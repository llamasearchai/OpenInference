"""
Continuous batching implementation for efficient LLM inference.
"""

import logging
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import uuid
import heapq

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class RequestState:
    """State tracking for a single inference request."""
    
    def __init__(self, 
                request_id: str,
                prompt_tokens: List[int],
                max_tokens: int,
                temperature: float = 1.0,
                top_p: float = 1.0,
                top_k: int = 0,
                callback: Optional[Callable[[List[int]], None]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize request state.
        
        Args:
            request_id: Unique ID for the request
            prompt_tokens: Input token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            callback: Optional callback for streaming tokens
            metadata: Optional request metadata
        """
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.callback = callback
        self.metadata = metadata or {}
        
        # State tracking
        self.is_prompt = True
        self.is_completed = False
        self.is_failed = False
        self.error_message = None
        
        # Token tracking
        self.generated_tokens = []
        self.total_generated = 0
        self.current_token = None
        
        # Position tracking
        self.prompt_pos = 0
        self.batch_idx = None  # Index in the current batch
        
        # Performance tracking
        self.creation_time = time.time()
        self.first_token_time = None
        self.last_token_time = None
        self.completion_time = None
    
    def get_next_token_to_process(self) -> Optional[int]:
        """
        Get the next token to process (either from prompt or previously generated).
        
        Returns:
            token: Next token ID or None if all tokens are processed
        """
        if self.is_prompt and self.prompt_pos < len(self.prompt_tokens):
            token = self.prompt_tokens[self.prompt_pos]
            self.prompt_pos += 1
            return token
        return None
    
    def add_generated_token(self, token_id: int):
        """
        Add a newly generated token.
        
        Args:
            token_id: Generated token ID
        """
        now = time.time()
        
        # Record first token time
        if self.first_token_time is None:
            self.first_token_time = now
            self.is_prompt = False
        
        self.current_token = token_id
        self.generated_tokens.append(token_id)
        self.total_generated += 1
        self.last_token_time = now
        
        # Call callback if provided
        if self.callback:
            self.callback([token_id])
    
    def mark_completed(self):
        """Mark the request as completed."""
        self.is_completed = True
        self.completion_time = time.time()
    
    def mark_failed(self, error_message: str):
        """Mark the request as failed."""
        self.is_failed = True
        self.error_message = error_message
        self.completion_time = time.time()
    
    def get_processing_time(self) -> float:
        """Get total processing time in seconds."""
        end_time = self.completion_time or time.time()
        return end_time - self.creation_time
    
    def get_tokens_per_second(self) -> float:
        """Get tokens per second generation rate."""
        if self.total_generated == 0:
            return 0.0
        
        if self.first_token_time is None:
            return 0.0
        
        end_time = self.completion_time or time.time()
        generation_time = end_time - self.first_token_time
        
        if generation_time <= 0:
            return 0.0
        
        return self.total_generated / generation_time
    
    def should_stop(self) -> bool:
        """Check if generation should stop for this request."""
        if self.is_completed or self.is_failed:
            return True
        
        if self.total_generated >= self.max_tokens:
            return True
        
        return False


class ContinuousBatcher:
    """
    Implements continuous batching for efficient LLM inference.
    
    Dynamically manages active requests, combining prompt prefill
    and token generation in a single batched loop that maximizes
    GPU utilization.
    """
    
    def __init__(self,
                model: Any,
                tokenizer: Any,
                max_batch_size: int = 32,
                max_input_length: int = 2048,
                max_output_length: int = 1024,
                max_active_requests: int = 256,
                prefill_batch_size: Optional[int] = None,
                decode_batch_size: Optional[int] = None,
                scheduler: str = "fifo",
                device: str = "cuda"):
        """
        Initialize continuous batcher.
        
        Args:
            model: Language model for inference
            tokenizer: Tokenizer for the model
            max_batch_size: Maximum overall batch size
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            max_active_requests: Maximum number of active requests
            prefill_batch_size: Batch size for prefill phase (None = auto)
            decode_batch_size: Batch size for decode phase (None = auto)
            scheduler: Scheduling algorithm ("fifo", "round_robin", or "priority")
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_active_requests = max_active_requests
        self.device = device
        self.scheduler = scheduler
        
        # Auto-configure batch sizes if not specified
        self.prefill_batch_size = prefill_batch_size or max(1, max_batch_size // 2)
        self.decode_batch_size = decode_batch_size or max_batch_size
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Request queues and state tracking
        self.waiting_requests = queue.Queue()
        self.active_requests = {}
        self.completed_requests = {}
        
        # For priority scheduling
        self.priority_queue = []
        
        # KV cache for active requests
        self.kv_caches = {}
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "active_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "tokens_processed": 0,
            "prefill_tokens_processed": 0,
            "decode_tokens_processed": 0,
            "prefill_batches": 0,
            "decode_batches": 0,
            "avg_batch_size": 0,
            "avg_tokens_per_second": 0,
            "avg_time_to_first_token": 0
        }
        
        # Threading for continuous inference
        self._stop_event = threading.Event()
        self._inference_thread = None
    
    def _schedule_requests(self, max_batch_size: int) -> List[str]:
        """
        Schedule next batch of requests based on scheduling algorithm.
        
        Args:
            max_batch_size: Maximum batch size to schedule
            
        Returns:
            request_ids: List of scheduled request IDs
        """
        scheduled = []
        
        if self.scheduler == "fifo":
            # Simple FIFO scheduling
            active_ids = list(self.active_requests.keys())
            return active_ids[:max_batch_size]
            
        elif self.scheduler == "round_robin":
            # Round-robin scheduler - rotate through active requests
            if not hasattr(self, "_last_scheduled"):
                self._last_scheduled = set()
            
            # Get all active requests not scheduled in the last round
            available = [rid for rid in self.active_requests if rid not in self._last_scheduled]
            
            # If all were scheduled in last round, reset
            if not available:
                available = list(self.active_requests.keys())
                self._last_scheduled = set()
            
            # Schedule up to max_batch_size
            scheduled = available[:max_batch_size]
            self._last_scheduled.update(scheduled)
            
            return scheduled
            
        elif self.scheduler == "priority":
            # Priority scheduling - maintain a priority queue
            # Priority determined by waiting time and custom priority
            now = time.time()
            
            # Build priority heap if empty
            if not self.priority_queue:
                for rid, req in self.active_requests.items():
                    wait_time = now - req.creation_time
                    priority = req.metadata.get("priority", 0)
                    heapq.heappush(self.priority_queue, (-priority, wait_time, rid))
            
            # Get top requests by priority
            scheduled = []
            for _ in range(min(max_batch_size, len(self.priority_queue))):
                if not self.priority_queue:
                    break
                _, _, rid = heapq.heappop(self.priority_queue)
                if rid in self.active_requests:
                    scheduled.append(rid)
            
            return scheduled
            
        else:
            # Default to FIFO
            active_ids = list(self.active_requests.keys())
            return active_ids[:max_batch_size]
    
    def start(self):
        """Start the continuous batching inference loop in a background thread."""
        if self._inference_thread is not None and self._inference_thread.is_alive():
            logger.warning("Inference thread is already running")
            return
        
        # Reset stop event
        self._stop_event.clear()
        
        # Start inference thread
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="ContinuousBatchingThread"
        )
        
        self._inference_thread.start()
        logger.info("Started continuous batching inference thread")
    
    def stop(self, wait: bool = True, timeout: Optional[float] = None):
        """
        Stop the continuous batching inference loop.
        
        Args:
            wait: Wait for the thread to stop
            timeout: Maximum time to wait in seconds
        """
        if self._inference_thread is None or not self._inference_thread.is_alive():
            logger.warning("Inference thread is not running")
            return
        
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to complete if requested
        if wait and self._inference_thread is not None:
            self._inference_thread.join(timeout=timeout)
            
            if self._inference_thread.is_alive():
                logger.warning("Inference thread did not stop within timeout")
            else:
                logger.info("Inference thread stopped successfully")
    
    def add_request(self,
                  prompt: Union[str, List[int]],
                  max_tokens: int = 128,
                  temperature: float = 1.0,
                  top_p: float = 1.0,
                  top_k: int = 0,
                  callback: Optional[Callable[[List[int]], None]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new request to the inference queue.
        
        Args:
            prompt: Input prompt text or token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            callback: Optional callback for streaming tokens
            metadata: Optional request metadata
            
        Returns:
            request_id: Unique ID for the request
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Tokenize prompt if it's a string
        prompt_tokens = prompt
        if isinstance(prompt, str):
            prompt_tokens = self.tokenizer.encode(prompt)
        
        # Check if prompt is too long
        if len(prompt_tokens) > self.max_input_length:
            raise ValueError(f"Prompt too long ({len(prompt_tokens)} tokens, max {self.max_input_length})")
        
        # Check if max_tokens is too large
        if max_tokens > self.max_output_length:
            raise ValueError(f"max_tokens too large ({max_tokens}, max {self.max_output_length})")
        
        # Create request state
        request = RequestState(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            callback=callback,
            metadata=metadata
        )
        
        # Add to queue
        self.waiting_requests.put(request)
        
        # Update metrics
        self.metrics["total_requests"] += 1
        
        logger.debug(f"Added request {request_id} with {len(prompt_tokens)} prompt tokens")
        return request_id
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get status of a request.
        
        Args:
            request_id: Request ID
            
        Returns:
            status: Request status information
        """
        # Check completed requests
        if request_id in self.completed_requests:
            request = self.completed_requests[request_id]
            return {
                "request_id": request_id,
                "status": "failed" if request.is_failed else "completed",
                "error": request.error_message,
                "prompt_tokens": len(request.prompt_tokens),
                "generated_tokens": request.total_generated,
                "output": request.generated_tokens,
                "processing_time": request.get_processing_time(),
                "tokens_per_second": request.get_tokens_per_second()
            }
            
        # Check active requests
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            return {
                "request_id": request_id,
                "status": "processing",
                "prompt_tokens": len(request.prompt_tokens),
                "generated_tokens": request.total_generated,
                "output": request.generated_tokens,
                "processing_time": request.get_processing_time(),
                "tokens_per_second": request.get_tokens_per_second()
            }
            
        # Request not found
        return {
            "request_id": request_id,
            "status": "unknown"
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        # Update metrics with current counts
        metrics = dict(self.metrics)
        metrics["active_requests"] = len(self.active_requests)
        metrics["waiting_requests"] = self.waiting_requests.qsize()
        
        # Calculate averages
        total_batches = metrics["prefill_batches"] + metrics["decode_batches"]
        if total_batches > 0:
            metrics["avg_batch_size"] = metrics["tokens_processed"] / total_batches
        
        # Return copy of metrics
        return metrics
    
    def _check_waiting_requests(self):
        """Check waiting requests queue and add to active requests if possible."""
        # Skip if we're at capacity
        if len(self.active_requests) >= self.max_active_requests:
            return
        
        # Calculate how many slots we have available
        available_slots = self.max_active_requests - len(self.active_requests)
        if available_slots <= 0:
            return
        
        # Move requests from waiting to active
        for _ in range(available_slots):
            try:
                request = self.waiting_requests.get_nowait()
                self.active_requests[request.request_id] = request
                logger.debug(f"Moved request {request.request_id} from waiting to active")
            except queue.Empty:
                break
    
    def _inference_loop(self):
        """Main inference loop for continuous batching."""
        logger.info("Starting continuous batching inference loop")
        
        with torch.no_grad():
            while not self._stop_event.is_set():
                try:
                    # Check for new requests
                    self._check_waiting_requests()
                    
                    # Skip if no active requests
                    if not self.active_requests:
                        time.sleep(0.01)
                        continue
                    
                    # First, handle any prefill requests
                    prefill_requests = {
                        rid: req for rid, req in self.active_requests.items() 
                        if req.is_prompt and not req.should_stop()
                    }
                    
                    if prefill_requests:
                        # Process a batch of prefill requests
                        batch_size = min(len(prefill_requests), self.prefill_batch_size)
                        prefill_request_ids = list(prefill_requests.keys())[:batch_size]
                        
                        self._process_prefill_batch(prefill_request_ids)
                    
                    # Then handle decode requests (token generation)
                    decode_requests = {
                        rid: req for rid, req in self.active_requests.items() 
                        if not req.is_prompt and not req.should_stop()
                    }
                    
                    if decode_requests:
                        # Schedule next batch of decode requests
                        batch_size = min(len(decode_requests), self.decode_batch_size)
                        scheduled_request_ids = self._schedule_requests(batch_size)
                        
                        # Process decode batch
                        if scheduled_request_ids:
                            self._process_decode_batch(scheduled_request_ids)
                    
                    # Clean up completed requests
                    self._cleanup_completed_requests()
                    
                    # Small sleep to avoid tight loop
                    if not prefill_requests and not decode_requests:
                        time.sleep(0.001)
                
                except Exception as e:
                    logger.error(f"Error in inference loop: {str(e)}", exc_info=True)
                    time.sleep(0.1)  # Avoid tight loop on error
        
        logger.info("Inference loop stopped")
    
    def _process_prefill_batch(self, request_ids: List[str]):
        """
        Process a batch of requests in prefill phase.
        
        Args:
            request_ids: List of request IDs to process
        """
        if not request_ids:
            return
        
        try:
            # Prepare input tensors
            batch_inputs = []
            max_length = 0
            request_objects = [self.active_requests[rid] for rid in request_ids]
            
            # Collect inputs and find max length
            for request in request_objects:
                if request.prompt_pos < len(request.prompt_tokens):
                    input_slice = request.prompt_tokens[:request.prompt_pos + 1]
                    batch_inputs.append(input_slice)
                    max_length = max(max_length, len(input_slice))
            
            # Skip if no inputs
            if not batch_inputs:
                return
            
            # Pad inputs to the same length
            padded_inputs = []
            attention_masks = []
            
            for input_ids in batch_inputs:
                padding_length = max_length - len(input_ids)
                padded_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                mask = [1] * len(input_ids) + [0] * padding_length
                
                padded_inputs.append(padded_ids)
                attention_masks.append(mask)
            
            # Convert to tensors
            input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=self.device)
            mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=self.device)
            
            # Run forward pass
            model_inputs = {
                "input_ids": input_tensor,
                "attention_mask": mask_tensor,
                "use_cache": True,
                "return_dict": True
            }
            
            outputs = self.model(**model_inputs)
            
            # Extract logits and KV cache
            logits = outputs.logits
            kv_cache = outputs.past_key_values
            
            # Store KV cache for each request
            for i, request_id in enumerate(request_ids):
                self.kv_caches[request_id] = tuple(
                    (k[:, i:i+1, :, :], v[:, i:i+1, :, :])
                    for k, v in kv_cache
                )
            
            # Process outputs for each request
            for i, request in enumerate(request_objects):
                # Get output for this request
                request_logits = logits[i, -1, :]
                
                # Sample next token
                next_token = self._sample_token(
                    request_logits,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k
                )
                
                # Add token to request
                request.add_generated_token(next_token.item())
            
            # Update metrics
            self.metrics["prefill_batches"] += 1
            self.metrics["prefill_tokens_processed"] += sum(len(req.prompt_tokens) for req in request_objects)
            self.metrics["tokens_processed"] += sum(len(req.prompt_tokens) for req in request_objects)
            
        except Exception as e:
            logger.error(f"Error in prefill batch: {str(e)}", exc_info=True)
            # Mark affected requests as failed
            for request_id in request_ids:
                if request_id in self.active_requests:
                    request = self.active_requests[request_id]
                    request.mark_failed(f"Prefill error: {str(e)}")
    
    def _process_decode_batch(self, request_ids: List[str]):
        """
        Process a batch of requests in decode (generation) phase.
        
        Args:
            request_ids: List of request IDs to process
        """
        if not request_ids:
            return
        
        try:
            # Prepare input tensors
            batch_inputs = []
            batch_caches = []
            request_objects = []
            valid_request_ids = []
            
            # Collect inputs and KV caches
            for request_id in request_ids:
                request = self.active_requests.get(request_id)
                kv_cache = self.kv_caches.get(request_id)
                
                # Skip if request or cache is missing
                if request is None or kv_cache is None or request.should_stop():
                    continue
                
                # Get the current token
                if request.current_token is None:
                    # This shouldn't happen for decode phase
                    logger.warning(f"Request {request_id} in decode phase with no current token")
                    continue
                
                # Add to batch
                batch_inputs.append([request.current_token])
                batch_caches.append(kv_cache)
                request_objects.append(request)
                valid_request_ids.append(request_id)
            
            # Skip if no valid requests
            if not batch_inputs:
                return
            
            # Convert inputs to tensor
            input_tensor = torch.tensor(batch_inputs, dtype=torch.long, device=self.device)
            
            # Prepare KV cache
            batch_size = len(batch_inputs)
            
            # Combine KV caches from all requests
            combined_kv_cache = []
            for layer_idx in range(len(batch_caches[0])):
                # Each layer has (key, value) pair
                k_tensors = [cache[layer_idx][0] for cache in batch_caches]
                v_tensors = [cache[layer_idx][1] for cache in batch_caches]
                
                # Concatenate on batch dimension
                k_combined = torch.cat(k_tensors, dim=1)
                v_combined = torch.cat(v_tensors, dim=1)
                
                combined_kv_cache.append((k_combined, v_combined))
            
            # Run forward pass
            model_inputs = {
                "input_ids": input_tensor,
                "past_key_values": combined_kv_cache,
                "use_cache": True,
                "return_dict": True
            }
            
            outputs = self.model(**model_inputs)
            
            # Extract logits and updated KV cache
            logits = outputs.logits
            kv_cache = outputs.past_key_values
            
            # Store updated KV cache for each request
            for i, request_id in enumerate(valid_request_ids):
                self.kv_caches[request_id] = tuple(
                    (k[:, i:i+1, :, :], v[:, i:i+1, :, :])
                    for k, v in kv_cache
                )
            
            # Process outputs for each request
            for i, request in enumerate(request_objects):
                # Get output for this request
                request_logits = logits[i, -1, :]
                
                # Sample next token
                next_token = self._sample_token(
                    request_logits,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k
                )
                
                # Add token to request
                request.add_generated_token(next_token.item())
                
                # Check if the request should stop (e.g., reached EOS token)
                if self._should_stop_generation(request, next_token.item()):
                    request.mark_completed()
            
            # Update metrics
            self.metrics["decode_batches"] += 1
            self.metrics["decode_tokens_processed"] += len(request_objects)
            self.metrics["tokens_processed"] += len(request_objects)
            
        except Exception as e:
            logger.error(f"Error in decode batch: {str(e)}", exc_info=True)
            # Mark affected requests as failed
            for request_id in valid_request_ids:
                if request_id in self.active_requests:
                    request = self.active_requests[request_id]
                    request.mark_failed(f"Decode error: {str(e)}")
    
    def _sample_token(self, 
                    logits: torch.Tensor, 
                    temperature: float = 1.0,
                    top_p: float = 1.0, 
                    top_k: int = 0) -> torch.Tensor:
        """
        Sample next token from logits.
        
        Args:
            logits: Token logits
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            token_id: Sampled token ID
        """
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # If temperature is 0, use greedy sampling
            return torch.argmax(logits, dim=-1, keepdim=True)
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Convert to probabilities and sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def _should_stop_generation(self, request: RequestState, token_id: int) -> bool:
        """
        Check if generation should stop for a request based on the new token.
        
        Args:
            request: Request state
            token_id: New token ID
            
        Returns:
            stop: Whether to stop generation
        """
        # Check if max tokens reached
        if request.total_generated >= request.max_tokens:
            return True
        
        # Check for EOS token
        if token_id == self.tokenizer.eos_token_id:
            return True
        
        # Check for custom stop tokens in metadata
        stop_tokens = request.metadata.get("stop_tokens", [])
        if token_id in stop_tokens:
            return True
        
        # Check for custom stop sequences
        stop_sequences = request.metadata.get("stop_sequences", [])
        for seq in stop_sequences:
            # Convert sequence to token IDs if it's a string
            if isinstance(seq, str):
                seq = self.tokenizer.encode(seq)
            
            # Check if the generated tokens end with this sequence
            if len(request.generated_tokens) >= len(seq):
                if request.generated_tokens[-len(seq):] == seq:
                    return True
        
        return False
    
    def _cleanup_completed_requests(self):
        """Move completed requests from active to completed state."""
        completed_ids = []
        
        # Find completed or failed requests
        for request_id, request in self.active_requests.items():
            if request.is_completed or request.is_failed or request.should_stop():
                if not request.is_completed and not request.is_failed:
                    request.mark_completed()
                
                completed_ids.append(request_id)
                self.completed_requests[request_id] = request
                
                # Update metrics
                if request.is_failed:
                    self.metrics["failed_requests"] += 1
                else:
                    self.metrics["completed_requests"] += 1
                
                # Clean up KV cache
                if request_id in self.kv_caches:
                    del self.kv_caches[request_id]
        
        # Remove from active requests
        for request_id in completed_ids:
            del self.active_requests[request_id]
    
    def wait_for_requests(self, request_ids: List[str], timeout: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """
        Wait for specific requests to complete.
        
        Args:
            request_ids: List of request IDs to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            results: Dictionary mapping request IDs to results
        """
        start_time = time.time()
        pending = set(request_ids)
        results = {}
        
        while pending:
            # Check if timeout reached
            if timeout is not None and time.time() - start_time > timeout:
                break
            
            # Check each request
            for request_id in list(pending):
                status = self.get_request_status(request_id)
                
                if status["status"] in ["completed", "failed"]:
                    results[request_id] = status
                    pending.remove(request_id)
            
            # Small sleep to avoid tight loop
            if pending:
                time.sleep(0.01)
        
        # Return results for all requests, including those still pending
        for request_id in pending:
            results[request_id] = self.get_request_status(request_id)
        
        return results