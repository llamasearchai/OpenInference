import time
import threading
import queue
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging
from dataclasses import dataclass, field

import torch
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TokenGenerationState:
    """State for token generation in a continuous batch."""
    
    request_id: str
    prompt: str
    generated_tokens: List[str] = field(default_factory=list)
    token_callback: Optional[Callable[[str, bool], None]] = None
    is_finished: bool = False
    start_time: float = field(default_factory=time.time)
    last_token_time: float = field(default_factory=time.time)
    
    # Generation parameters
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    
    # Sequence tracking
    input_token_ids: Optional[torch.Tensor] = None
    current_token_position: int = 0
    
    def add_token(self, token: str):
        """Add a generated token."""
        self.generated_tokens.append(token)
        self.last_token_time = time.time()
        self.current_token_position += 1
        
        # Call the token callback if provided
        if self.token_callback:
            is_finished = self.is_finished or (self.current_token_position >= self.max_length)
            try:
                if isinstance(self.token_callback, queue.Queue):
                    self.token_callback.put((token, is_finished))
                else:
                    self.token_callback(token, is_finished)
            except Exception as e:
                logger.error(f"Error in token callback: {str(e)}")
    
    def finish(self):
        """Mark generation as finished."""
        self.is_finished = True
        
        # Call the token callback with finished=True if provided
        if self.token_callback:
            try:
                if isinstance(self.token_callback, queue.Queue):
                    self.token_callback.put(("", True)) # Sentinel for completion
                else:
                    # Ensure even non-queue callbacks are notified of final finish
                    if not self.generated_tokens or self.current_token_position < self.max_length:
                         self.token_callback("", True)
            except Exception as e:
                logger.error(f"Error in token callback: {str(e)}")
    
    def get_total_tokens(self) -> int:
        """Get the total number of tokens (input + generated)."""
        return (len(self.input_token_ids) if self.input_token_ids is not None else 0) + len(self.generated_tokens)
    
    def get_generated_text(self) -> str:
        """Get the generated text as a string."""
        return "".join(self.generated_tokens)


class ContinuousBatcher:
    """
    Continuous batching system for LLM token generation.
    
    Optimizes throughput by processing requests in a continuous batch,
    adding new requests and removing completed ones dynamically.
    """
    
    def __init__(self, 
                 model_fn: Callable,
                 tokenizer: Any,
                 max_batch_size: int = 8,
                 max_sequence_length: int = 2048,
                 device: str = "cuda"):
        """
        Initialize the continuous batcher.
        
        Args:
            model_fn: LLM model forward function
            tokenizer: Tokenizer for the model
            max_batch_size: Maximum batch size
            max_sequence_length: Maximum sequence length
            device: Device to run inference on
        """
        self.model_fn = model_fn
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.device = device
        
        # Queue for incoming requests
        self.request_queue = queue.Queue()
        
        # Active generation states
        self.active_states = {}
        
        # Processing thread
        self.worker_thread = None
        self.is_running = False
        
        # KV cache manager
        from ..runtime.memory_manager import KVCacheManager
        self.kv_cache_manager = KVCacheManager(
            max_seq_length=max_sequence_length,
            max_batch_size=max_batch_size,
            device=device
        )
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "active_requests": 0,
            "completed_requests": 0,
            "total_tokens_generated": 0,
            "errors": 0
        }
    
    def start(self):
        """Start the continuous batcher."""
        if self.is_running:
            return
            
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Continuous batcher started")
    
    def stop(self):
        """Stop the continuous batcher."""
        self.is_running = False
        
        # Finish all active generations
        for state in list(self.active_states.values()):
            state.finish()
        
        # Clean up KV cache
        self.kv_cache_manager.cleanup()
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
            
        logger.info("Continuous batcher stopped")
    
    def submit(self, prompt: str, callback: Any, **generation_params):
        """
        Submit a prompt for generation.
        
        Args:
            prompt: Text prompt to generate from
            callback: Either a function(text, is_finished) for streaming or 
                     function(result, success) for completion
            **generation_params: Parameters for generation (max_length, temperature, etc.)
        """
        # Create a unique request ID
        request_id = f"req_{time.time()}_{id(prompt)}"
        
        # Determine if this is a streaming or completion request
        is_streaming = False
        if callable(callback):
            try:
                import inspect
                sig = inspect.signature(callback)
                # If callback accepts two positional args and the second is "is_finished", 
                # it's likely a streaming callback
                params = list(sig.parameters.values())
                if len(params) >= 2:
                    is_streaming = True
            except Exception:
                # If we can't inspect, assume completion
                pass
        
        # Create a token generation state
        state = TokenGenerationState(
            request_id=request_id,
            prompt=prompt,
            token_callback=callback if is_streaming else None,
            max_length=generation_params.get("max_length", 512),
            temperature=generation_params.get("temperature", 1.0),
            top_p=generation_params.get("top_p", 0.9),
            top_k=generation_params.get("top_k", 50)
        )
        
        # For completion requests, wrap the callback
        if not is_streaming:
            original_callback = callback
            state.token_callback = lambda token, is_finished: None  # Collect tokens but don't call back until complete
        
        # Tokenize the prompt
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            state.input_token_ids = input_ids
        except Exception as e:
            logger.error(f"Error tokenizing prompt: {str(e)}")
            if not is_streaming and original_callback:
                original_callback(None, False)  # Signal failure
            return
        
        # Add to request queue
        self.request_queue.put(state)
        
        # Update statistics
        self.stats["total_requests"] += 1
        
        logger.debug(f"Submitted generation request {request_id}")
    
    def _worker_loop(self):
        """Worker loop for continuous batching."""
        while self.is_running:
            try:
                # Process a batch of tokens
                self._process_batch()
                
                # Brief sleep to avoid spinning
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in continuous batcher worker: {str(e)}")
                self.stats["errors"] += 1
    
    def _process_batches(self):
        """Main worker loop to process batches of requests with improved reliability."""
        if torch is None:  # Replaced TORCH_AVAILABLE check
            logger.error("PyTorch not available, cannot process batches")
            return
        
        # This outer try-except is for fatal errors that might require restarting the worker.
        try:
            # Set model to eval mode and configure for inference
            # Assuming self.model exists and has an eval method
            if not hasattr(self, 'model') or not callable(getattr(self.model, 'eval', None)):
                logger.error("Model is not properly initialized or does not have an eval method.")
                return

            self.model.eval()
            
            # Configure torch settings for inference
            if hasattr(torch, 'inference_mode'):
                inference_context = torch.inference_mode()
            else:
                inference_context = torch.no_grad()
            
            # Assuming self.stop_event and self.lock are initialized
            if not hasattr(self, 'stop_event') or not hasattr(self, 'lock'):
                logger.error("stop_event or lock not initialized in ContinuousBatcher.")
                return

            with inference_context:
                while not self.stop_event.is_set():
                    # This inner try-except handles errors within a single batch processing iteration.
                    try:
                        # Handle new requests
                        new_requests = []
                        try:
                            # Get new requests with timeout
                            while not self.request_queue.empty() and len(new_requests) < self.max_batch_size:
                                try:
                                    request = self.request_queue.get_nowait()
                                    new_requests.append(request)
                                except queue.Empty:
                                    break
                        except Exception as e: # More specific exception if possible
                            logger.warning(f"Error getting new requests: {e}")
                        
                        # Process new prompts if any
                        if new_requests:
                            if not hasattr(self, '_process_new_prompts') or not callable(self._process_new_prompts):
                                logger.error("_process_new_prompts method not found.")
                            else:
                                self._process_new_prompts(new_requests)
                        
                        # Get active (unfinished) requests
                        active_requests = []
                        with self.lock: # Ensure self.lock is properly initialized
                            # Assuming self.active_requests is a dict of objects/dicts with "is_finished"
                            active_requests = [r for r in self.active_states.values() if not r.get("is_finished") and not getattr(r, 'is_finished', True)]


                        # Process the next token generation for active requests
                        if active_requests:
                            if not hasattr(self, '_process_next_token_batch') or not callable(self._process_next_token_batch):
                                logger.error("_process_next_token_batch method not found.")
                            else:
                                self._process_next_token_batch(active_requests)
                        else:
                            # No active requests, sleep briefly to prevent CPU spinning
                            time.sleep(0.005)
                        
                        # Add a forced garbage collection every N iterations to prevent memory fragmentation
                        if hasattr(self, '_gc_counter'):
                            self._gc_counter += 1
                            if self._gc_counter > 1000:  # Every ~1000 iterations
                                self._gc_counter = 0
                                import gc
                                gc.collect()
                        else:
                            self._gc_counter = 0
                    
                    except Exception as e:
                        logger.error(f"Error in batch processing loop: {str(e)}")
                        # Don't crash the loop - continue with the next iteration
                        time.sleep(0.1)  # Brief pause to prevent tight error loops
        
        except Exception as e: # This corresponds to the outer try
            logger.error(f"Fatal error in continuous batcher worker: {str(e)}")
            # Try to restart the worker
            time.sleep(1.0)
            try:
                # Clear any pending requests to prevent processing old data
                while not self.request_queue.empty():
                    try:
                        self.request_queue.get_nowait()
                    except queue.Empty:
                        break
                    
                # Launch a new worker thread
                if not self.stop_event.is_set():
                    self.worker_thread = threading.Thread(
                        target=self._process_batches,
                        name="ContinuousBatcher-Worker"
                    )
                    self.worker_thread.daemon = True
                    self.worker_thread.start()
                    logger.info("Continuous batcher worker thread restarted after error")
            except Exception as restart_e: # More specific exception if possible
                logger.error(f"Failed to restart worker thread: {str(restart_e)}")
    
    def _add_new_requests(self):
        """Add new requests from the queue to active states."""
        # Add new requests up to max_batch_size
        while len(self.active_states) < self.max_batch_size and not self.request_queue.empty():
            try:
                # Get next request
                state = self.request_queue.get_nowait()
                
                # Add to active states
                self.active_states[state.request_id] = state
                
                # Allocate KV cache
                self.kv_cache_manager.allocate_cache(state.request_id, batch_size=1)
                
                # Update statistics
                self.stats["active_requests"] = len(self.active_states)
                
                logger.debug(f"Added request {state.request_id} to active batch")
            except queue.Empty:
                break
    
    def _prepare_batch(self):
        """
        Prepare input batch for model inference.
        
        Returns:
            batch_input_ids: Tensor of input IDs
            batch_attention_mask: Tensor of attention masks
            batch_state_map: Mapping from batch indices to states
        """
        if not self.active_states:
            return None, None, None
            
        # Prepare batch
        batch_input_ids = []
        batch_attention_mask = []
        batch_state_map = {}
        
        # For each active state
        for batch_idx, (request_id, state) in enumerate(list(self.active_states.items())):
            try:
                if state.is_finished:
                    # Remove finished states
                    del self.active_states[request_id]
                    self.kv_cache_manager.release_cache(request_id)
                    self.stats["completed_requests"] += 1
                    self.stats["active_requests"] = len(self.active_states)
                    continue
                
                if state.current_token_position == 0:
                    # First token generation, use the entire prompt
                    input_ids = state.input_token_ids
                else:
                    # For subsequent tokens, we just need the previously generated token
                    input_ids = torch.tensor([[self.tokenizer.convert_tokens_to_ids(state.generated_tokens[-1])]], 
                                            device=self.device)
                
                # Add to batch
                batch_input_ids.append(input_ids)
                
                # Create attention mask (all 1s for now)
                attention_mask = torch.ones_like(input_ids)
                batch_attention_mask.append(attention_mask)
                
                # Map batch index to state
                batch_state_map[batch_idx] = state
                
            except Exception as e:
                logger.error(f"Error preparing batch for request {request_id}: {str(e)}")
                state.finish()
                del self.active_states[request_id]
                self.kv_cache_manager.release_cache(request_id)
                self.stats["errors"] += 1
        
        if not batch_input_ids:
            return None, None, None
        
        # Stack tensors
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_attention_mask = torch.cat(batch_attention_mask, dim=0)
        
        return batch_input_ids, batch_attention_mask, batch_state_map
    
    def _sample_next_tokens(self, logits, batch_state_map):
        """
        Sample next tokens from logits.
        
        Args:
            logits: Next token logits from model
            batch_state_map: Mapping from batch indices to states
            
        Returns:
            next_tokens: List of sampled tokens
        """
        next_tokens = []
        
        for batch_idx, state in batch_state_map.items():
            try:
                # Get logits for this sequence
                seq_logits = logits[batch_idx]
                
                # Apply temperature
                if state.temperature > 0:
                    seq_logits = seq_logits / state.temperature
                
                # Apply top-k sampling
                if state.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(seq_logits, state.top_k)
                    
                    # Create a mask of selected tokens
                    mask = torch.zeros_like(seq_logits)
                    mask.scatter_(0, top_k_indices, 1)
                    
                    # Apply mask
                    seq_logits = torch.where(mask.bool(), seq_logits, torch.tensor(-float('inf'), device=seq_logits.device))
                
                # Apply top-p (nucleus) sampling
                if 0 < state.top_p < 1.0:
                    # Sort logits
                    sorted_logits, sorted_indices = torch.sort(seq_logits, descending=True)
                    
                    # Calculate cumulative probabilities
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Create mask for tokens above threshold
                    sorted_indices_to_remove = cumulative_probs > state.top_p
                    sorted_indices_to_remove[0] = False  # Keep at least one token
                    
                    # Shift the mask to the right to keep tokens with cumulative probability < p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    
                    # Create indices mask
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    
                    # Apply mask
                    seq_logits = seq_logits.masked_fill(indices_to_remove, -float('inf'))
                
                # Convert to probabilities
                probs = torch.softmax(seq_logits, dim=-1)
                
                # Sample next token
                next_token_id = torch.multinomial(probs, 1).item()
                
                # Convert to token
                next_token = self.tokenizer.decode([next_token_id])
                
                next_tokens.append(next_token)
                
            except Exception as e:
                logger.error(f"Error sampling next token: {str(e)}")
                next_tokens.append("")
        
        return next_tokens
    
    def _process_generated_tokens(self, next_tokens, batch_state_map):
        """
        Process generated tokens for each state.
        
        Args:
            next_tokens: List of generated tokens
            batch_state_map: Mapping from batch indices to states
        """
        for batch_idx, state in batch_state_map.items():
            try:
                token = next_tokens[batch_idx]
                
                if not token:
                    # Empty token, mark as finished
                    state.finish()
                    continue
                
                # Add token to state
                state.add_token(token)
                
                # Update statistics
                self.stats["total_tokens_generated"] += 1
                
                # Check if we've reached max length
                if state.current_token_position >= state.max_length:
                    state.finish()
                
                # Check for EOS token
                if token == self.tokenizer.eos_token:
                    state.finish()
                
            except Exception as e:
                logger.error(f"Error processing generated token: {str(e)}")
                state.finish()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        stats = self.stats.copy()
        
        # Add throughput calculations
        if stats["total_tokens_generated"] > 0:
            total_time = sum(time.time() - state.start_time for state in self.active_states.values())
            if total_time > 0:
                stats["tokens_per_second"] = stats["total_tokens_generated"] / total_time
            else:
                stats["tokens_per_second"] = 0
        else:
            stats["tokens_per_second"] = 0
        
        # Add queue size
        stats["queue_size"] = self.request_queue.qsize()
        
        # Add active batch utilization
        stats["batch_utilization"] = len(self.active_states) / max(1, self.max_batch_size)
        
        # Add KV cache stats
        stats["kv_cache"] = self.kv_cache_manager.get_stats()
        
        return stats