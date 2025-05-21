"""
Continuous batching implementation for high-throughput inference.
"""

import logging
import time
import threading
import queue
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

@dataclass
class TokenRequest:
    """Request for token generation."""
    request_id: str
    prompt_tokens: List[int]
    max_new_tokens: int
    callback: Callable[[List[int], bool], None]
    params: Dict[str, Any]
    created_at: float = 0.0
    last_updated_at: float = 0.0

@dataclass
class TokenResponse:
    """Response with generated tokens."""
    request_id: str
    generated_tokens: List[int]
    is_done: bool
    token_count: int = 0

class ContinuousBatcher:
    """
    Continuous batching for transformer inference.
    
    Implements prefill and decode batching with continuous token streaming
    for high-throughput inference.
    """
    
    def __init__(self,
                model: Any,
                max_batch_size: int = 32,
                max_input_length: int = 2048,
                max_prefill_tokens: int = 4096,
                max_attention_window: int = 4096,
                max_kv_cache_entries: int = 4096 * 32,
                device: str = "cuda"):
        """
        Initialize the continuous batcher.
        
        Args:
            model: Model to use for inference
            max_batch_size: Maximum batch size
            max_input_length: Maximum input sequence length
            max_prefill_tokens: Maximum tokens to process in prefill phase
            max_attention_window: Maximum attention window size
            max_kv_cache_entries: Maximum KV cache entries
            device: Device to run on
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_input_length = max_input_length
        self.max_prefill_tokens = max_prefill_tokens
        self.max_attention_window = max_attention_window
        self.max_kv_cache_entries = max_kv_cache_entries
        self.device = device
        
        # Queues
        self.prefill_queue = queue.Queue()
        self.decode_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Request tracking
        self.active_requests = {}  # request_id -> TokenRequest
        self.request_to_batch = {}  # request_id -> batch_id
        self.pending_prefill = []  # List of requests awaiting prefill
        self.pending_decode = []  # List of requests awaiting decode
        
        # Batch tracking
        self.active_batches = {}  # batch_id -> {request_ids, state, kv_caches}
        self.next_batch_id = 0
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_prefill_batches": 0,
            "total_decode_batches": 0,
            "total_tokens_generated": 0,
            "prefill_tokens_per_second": 0,
            "decode_tokens_per_second": 0,
            "average_batch_size": 0,
            "average_request_time_ms": 0,
            "active_requests": 0,
            "queue_depth": 0
        }
        
        # Control flags
        self.running = False
        self.stop_event = threading.Event()
        
        # Worker threads
        self.prefill_thread = None
        self.decode_thread = None
        self.response_thread = None
    
    def start(self):
        """Start the batching worker threads."""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Start worker threads
        self.prefill_thread = threading.Thread(target=self._prefill_worker, daemon=True)
        self.decode_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self.response_thread = threading.Thread(target=self._response_worker, daemon=True)
        
        self.prefill_thread.start()
        self.decode_thread.start()
        self.response_thread.start()
        
        logger.info(f"Continuous batcher started with max batch size {self.max_batch_size}")
    
    def stop(self):
        """Stop the batching worker threads."""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.prefill_thread and self.prefill_thread.is_alive():
            self.prefill_thread.join(timeout=2.0)
        
        if self.decode_thread and self.decode_thread.is_alive():
            self.decode_thread.join(timeout=2.0)
        
        if self.response_thread and self.response_thread.is_alive():
            self.response_thread.join(timeout=2.0)
        
        logger.info("Continuous batcher stopped")
    
    def submit_request(self, 
                      prompt_tokens: List[int],
                      max_new_tokens: int,
                      callback: Callable[[List[int], bool], None],
                      **params) -> str:
        """
        Submit a new request for generation.
        
        Args:
            prompt_tokens: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            callback: Callback function for token streaming
            **params: Additional generation parameters
            
        Returns:
            request_id: Unique ID for this request
        """
        # Create unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Create request object
        request = TokenRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            callback=callback,
            params=params,
            created_at=time.time(),
            last_updated_at=time.time()
        )
        
        # Store in active requests
        self.active_requests[request_id] = request
        
        # Add to prefill queue
        self.prefill_queue.put(request)
        
        # Update stats
        self.stats["total_requests"] += 1
        self.stats["active_requests"] += 1
        self.stats["queue_depth"] = self.prefill_queue.qsize()
        
        logger.debug(f"Submitted request {request_id} with {len(prompt_tokens)} prompt tokens")
        
        return request_id
    
    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel an active request.
        
        Args:
            request_id: Request ID to cancel
            
        Returns:
            success: Whether cancellation was successful
        """
        if request_id not in self.active_requests:
            logger.warning(f"Cannot cancel non-existent request {request_id}")
            return False
        
        # Remove from active requests
        if request_id in self.active_requests:
            request = self.active_requests.pop(request_id)
            
            # Call callback with done flag
            try:
                request.callback([], True)
            except Exception as e:
                logger.error(f"Error in callback for cancelled request {request_id}: {e}")
        
        # Remove from batch if part of one
        if request_id in self.request_to_batch:
            batch_id = self.request_to_batch.pop(request_id)
            if batch_id in self.active_batches:
                if request_id in self.active_batches[batch_id]["request_ids"]:
                    self.active_batches[batch_id]["request_ids"].remove(request_id)
                
                # If batch is now empty, remove it
                if not self.active_batches[batch_id]["request_ids"]:
                    self.active_batches.pop(batch_id)
        
        # Update stats
        self.stats["active_requests"] -= 1
        
        logger.debug(f"Cancelled request {request_id}")
        return True
    
    def _prefill_worker(self):
        """Worker thread that processes prefill batches."""
        logger.info("Prefill worker started")
        
        while not self.stop_event.is_set():
            try:
                # Collect requests for batching
                current_batch = []
                current_batch_tokens = 0
                
                # Get the first request (blocking)
                try:
                    request = self.prefill_queue.get(timeout=0.1)
                    current_batch.append(request)
                    current_batch_tokens = len(request.prompt_tokens)
                except queue.Empty:
                    continue
                
                # Try to fill the batch with more requests (non-blocking)
                batch_timeout = time.time() + 0.005  # 5ms batch collection window
                while (len(current_batch) < self.max_batch_size and 
                       current_batch_tokens < self.max_prefill_tokens and
                       time.time() < batch_timeout):
                    try:
                        request = self.prefill_queue.get_nowait()
                        # Check if adding this would exceed limits
                        if (current_batch_tokens + len(request.prompt_tokens) <= self.max_prefill_tokens):
                            current_batch.append(request)
                            current_batch_tokens += len(request.prompt_tokens)
                        else:
                            # Put back and process in next batch
                            self.prefill_queue.put(request)
                            break
                    except queue.Empty:
                        break
                
                # Process the batch
                if current_batch:
                    batch_id = self._process_prefill_batch(current_batch)
                    
                    # Update stats
                    self.stats["total_prefill_batches"] += 1
                    self.stats["average_batch_size"] = (
                        (self.stats["average_batch_size"] * (self.stats["total_prefill_batches"] - 1) +
                         len(current_batch)) / self.stats["total_prefill_batches"]
                    )
                    self.stats["queue_depth"] = self.prefill_queue.qsize()
                
            except Exception as e:
                logger.error(f"Error in prefill worker: {str(e)}", exc_info=True)
                time.sleep(0.1)  # Avoid tight loop on error
    
    def _process_prefill_batch(self, requests: List[TokenRequest]) -> int:
        """
        Process a batch of requests in the prefill phase.
        
        Args:
            requests: List of requests to process
            
        Returns:
            batch_id: ID of the created batch
        """
        start_time = time.time()
        
        # Create a new batch
        batch_id = self.next_batch_id
        self.next_batch_id += 1
        
        # Initialize batch state
        self.active_batches[batch_id] = {
            "request_ids": [request.request_id for request in requests],
            "state": "prefill",
            "kv_cache": None,
            "attention_mask": None,
            "input_lengths": {},
            "generated_tokens": {},
            "created_at": start_time
        }
        
        # Update request-to-batch mapping
        for request in requests:
            self.request_to_batch[request.request_id] = batch_id
            self.active_batches[batch_id]["input_lengths"][request.request_id] = len(request.prompt_tokens)
            self.active_batches[batch_id]["generated_tokens"][request.request_id] = []
        
        try:
            # Prepare inputs for the model
            # This would be different based on your model's input format
            import torch
            
            # Get max length in this batch for padding
            max_length = max(len(request.prompt_tokens) for request in requests)
            
            # Prepare input tensors
            input_ids = torch.zeros((len(requests), max_length), dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((len(requests), max_length), dtype=torch.long, device=self.device)
            
            # Fill in the actual token values
            for i, request in enumerate(requests):
                tokens = request.prompt_tokens
                seq_len = len(tokens)
                input_ids[i, :seq_len] = torch.tensor(tokens, dtype=torch.long, device=self.device)
                attention_mask[i, :seq_len] = 1
            
            # Run model forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True
                )
            
            # Extract logits and KV cache
            logits = outputs.logits
            kv_cache = outputs.past_key_values
            
            # Store KV cache in batch state
            self.active_batches[batch_id]["kv_cache"] = kv_cache
            self.active_batches[batch_id]["attention_mask"] = attention_mask
            
            # Process output logits to get next token for each request
            next_token_logits = logits[:, -1, :]
            
            # Apply sampling/generation parameters for each request
            for i, request in enumerate(requests):
                # Get generation parameters for this request
                temperature = request.params.get("temperature", 1.0)
                top_p = request.params.get("top_p", 1.0)
                top_k = request.params.get("top_k", 0)
                do_sample = request.params.get("do_sample", temperature > 0.0)
                
                # Get logits for this request
                req_logits = next_token_logits[i].unsqueeze(0)
                
                # Apply temperature
                if temperature > 0:
                    req_logits = req_logits / temperature
                
                # Apply top-k if specified
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(req_logits, k=min(top_k, req_logits.shape[-1]))
                    req_logits.fill_(-float('Inf'))
                    req_logits.scatter_(1, top_k_indices, top_k_values)
                
                # Apply top-p if specified
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(req_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    req_logits[indices_to_remove] = -float('Inf')
                
                # Sample or argmax
                if do_sample:
                    probs = torch.softmax(req_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(req_logits, dim=-1).item()
                
                # Add token to generated tokens list
                self.active_batches[batch_id]["generated_tokens"][request.request_id].append(next_token)
                
                # Create response
                response = TokenResponse(
                    request_id=request.request_id,
                    generated_tokens=[next_token],
                    is_done=False,
                    token_count=1
                )
                
                # Add to response queue
                self.response_queue.put(response)
            
            # Move batch to decode phase
            self.active_batches[batch_id]["state"] = "decode"
            
            # Add to decode queue with priority based on batch size (smaller batches first)
            # This helps maintain interactivity for single requests
            self.decode_queue.put((batch_id, -len(requests)))
            
            # Calculate tokens per second
            elapsed = time.time() - start_time
            tokens_processed = sum(len(request.prompt_tokens) for request in requests)
            if elapsed > 0:
                throughput = tokens_processed / elapsed
                # Update stats with exponential moving average
                self.stats["prefill_tokens_per_second"] = (
                    self.stats["prefill_tokens_per_second"] * 0.95 + throughput * 0.05
                )
            
            logger.debug(f"Processed prefill batch {batch_id} with {len(requests)} requests in {elapsed:.3f}s")
            
            return batch_id
            
        except Exception as e:
            logger.error(f"Error processing prefill batch: {str(e)}", exc_info=True)
            
            # Handle error by notifying clients
            for request in requests:
                if request.request_id in self.active_requests:
                    try:
                        # Notify callback of error
                        request.callback([], True)
                    except Exception as cb_error:
                        logger.error(f"Error in callback for {request.request_id}: {str(cb_error)}")
                    
                    # Clean up request
                    self.active_requests.pop(request.request_id, None)
                    self.request_to_batch.pop(request.request_id, None)
                    self.stats["active_requests"] -= 1
            
            # Clean up batch
            self.active_batches.pop(batch_id, None)
            
            return -1
    
    def _decode_worker(self):
        """Worker thread that processes decode batches."""
        logger.info("Decode worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get batch from queue
                try:
                    batch_id, _ = self.decode_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Check if batch exists
                if batch_id not in self.active_batches:
                    logger.warning(f"Batch {batch_id} not found in active batches")
                    continue
                
                # Get batch info
                batch = self.active_batches[batch_id]
                
                # Verify we're in decode phase
                if batch["state"] != "decode":
                    logger.warning(f"Batch {batch_id} not in decode phase")
                    continue
                
                # Process one decode step
                self._process_decode_step(batch_id)
                
            except Exception as e:
                logger.error(f"Error in decode worker: {str(e)}", exc_info=True)
                time.sleep(0.1)  # Avoid tight loop on error
    
    def _process_decode_step(self, batch_id: int):
        """
        Process one step of decoding for a batch.
        
        Args:
            batch_id: Batch ID to process
        """
        start_time = time.time()
        
        # Check if batch still exists, it might have been cleaned up
        if batch_id not in self.active_batches:
            logger.warning(f"Decode step called for non-existent or cleaned up batch {batch_id}")
            return

        batch = self.active_batches[batch_id]
        
        # Get active requests in this batch
        # Filter request_ids first to ensure they are still in self.active_requests
        current_request_ids_in_batch = [req_id for req_id in batch["request_ids"] if req_id in self.active_requests]
        active_requests = [self.active_requests[req_id] for req_id in current_request_ids_in_batch]
        
        # Skip if no active requests or batch was somehow emptied
        if not active_requests:
            logger.debug(f"No active requests in batch {batch_id} during decode step, removing batch.")
            self.active_batches.pop(batch_id, None)
            # Also clean up request_to_batch mappings for these potentially orphaned ids
            for req_id_original in batch["request_ids"]:
                 if self.request_to_batch.get(req_id_original) == batch_id:
                    self.request_to_batch.pop(req_id_original, None)
            return
        
        # Update the batch's request_ids to only contain currently active ones for this step
        batch["request_ids"] = current_request_ids_in_batch

        try:
            import torch
            
            # Prepare input for decoder step
            batch_size = len(active_requests)
            
            # Get the last generated token for each request
            input_ids = torch.tensor(
                [[batch["generated_tokens"][req.request_id][-1]] for req in active_requests],
                dtype=torch.long, device=self.device
            )
            
            # Get attention mask
            attention_mask = batch["attention_mask"]
            
            # For requests that were removed, we need to filter the attention mask and KV cache
            if len(active_requests) < attention_mask.shape[0]:
                # Find indices of active requests
                active_indices = []
                for i, req_id in enumerate(current_request_ids_in_batch):
                    if req_id in self.active_requests:
                        active_indices.append(i)
                
                # Filter attention mask
                attention_mask = attention_mask[active_indices]
                
                # Filter KV cache
                filtered_kv_cache = []
                for layer_past in batch["kv_cache"]:
                    # For each layer, extract the active requests' KV tensors
                    filtered_layer_past = tuple(tensor[active_indices] for tensor in layer_past)
                    filtered_kv_cache.append(filtered_layer_past)
                
                batch["kv_cache"] = tuple(filtered_kv_cache)
                
                # Update request IDs list
                batch["request_ids"] = current_request_ids_in_batch
                current_request_ids_in_batch = batch["request_ids"]
            
            # Extend attention mask for new token
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((len(active_requests), 1), dtype=torch.long, device=self.device)
            ], dim=1)
            
            # Run model forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=batch["kv_cache"],
                    use_cache=True,
                    return_dict=True
                )
            
            # Extract logits and updated KV cache
            logits = outputs.logits
            batch["kv_cache"] = outputs.past_key_values
            batch["attention_mask"] = attention_mask
            
            # Process output logits to get next token for each request
            next_token_logits = logits[:, -1, :]
            
            # Track finished requests to remove from batch
            finished_requests = []
            
            # Apply sampling/generation parameters for each request
            for i, request in enumerate(active_requests):
                req_id = request.request_id
                
                # Get generation parameters
                temperature = request.params.get("temperature", 1.0)
                top_p = request.params.get("top_p", 1.0)
                top_k = request.params.get("top_k", 0)
                do_sample = request.params.get("do_sample", temperature > 0.0)
                stop_tokens = request.params.get("stop_tokens", [])
                
                # Get logits for this request
                req_logits = next_token_logits[i].unsqueeze(0)
                
                # Apply temperature
                if temperature > 0:
                    req_logits = req_logits / temperature
                
                # Apply top-k if specified
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(req_logits, k=min(top_k, req_logits.shape[-1]))
                    req_logits.fill_(-float('Inf'))
                    req_logits.scatter_(1, top_k_indices, top_k_values)
                
                # Apply top-p if specified
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(req_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    req_logits[indices_to_remove] = -float('Inf')
                
                # Sample or argmax
                if do_sample:
                    probs = torch.softmax(req_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(req_logits, dim=-1).item()
                
                # Add token to generated tokens list
                batch["generated_tokens"][req_id].append(next_token)
                
                # Create response
                response = TokenResponse(
                    request_id=req_id,
                    generated_tokens=[next_token],
                    is_done=False, # This will be updated later if EOS or max_length
                    token_count=len(batch["generated_tokens"][req_id])
                )
                
                # Add to response queue
                self.response_queue.put(response)

                # Check if this request is now finished
                # (e.g., EOS token, max_new_tokens reached)
                # This logic might need to be more sophisticated, checking for EOS from tokenizer etc.
                if len(batch["generated_tokens"][req_id]) >= request.max_new_tokens or next_token == getattr(self.model.tokenizer, 'eos_token_id', -1): # Assuming tokenizer is accessible
                    finished_requests.append(req_id)
                    # Update response to indicate done
                    response.is_done = True # We can modify the response object before it's picked by another thread

            # Remove finished requests from this batch for future decode steps
            if finished_requests:
                batch["request_ids"] = [req_id for req_id in batch["request_ids"] if req_id not in finished_requests]
                for req_id in finished_requests:
                    if req_id in self.active_requests:
                        # Call the callback for the final time with is_done=True
                        try:
                            self.active_requests[req_id].callback(batch["generated_tokens"][req_id], True)
                        except Exception as e:
                            logger.error(f"Error in final callback for request {req_id}: {e}")
                        # Move to completed requests
                        self.active_requests.pop(req_id) 
                        self.request_to_batch.pop(req_id, None)
                        self.stats["active_requests"] -=1

            # If batch is now empty, remove it from active_batches
            if not batch["request_ids"]:
                logger.debug(f"Batch {batch_id} is now empty, removing.")
                self.active_batches.pop(batch_id, None)
            else:
                # Otherwise, put it back in the decode_queue for next token generation
                self.decode_queue.put((batch_id, -len(batch["request_ids"]))) # Re-queue with updated priority

            # Calculate tokens per second for this decode step
            elapsed = time.time() - start_time
            tokens_processed = len(active_requests) # Each active request generates one token
            if elapsed > 0:
                throughput = tokens_processed / elapsed
                self.stats["decode_tokens_per_second"] = (
                    self.stats["decode_tokens_per_second"] * 0.95 + throughput * 0.05
                )
            
            logger.debug(f"Processed decode step for batch {batch_id} with {len(active_requests)} requests in {elapsed:.3f}s")

        except Exception as e:
            logger.error(f"Error processing decode step for batch {batch_id}: {str(e)}", exc_info=True)
            # Handle error by notifying clients for all requests in this batch attempt
            for req_id_in_error in batch.get("request_ids", []):
                 if req_id_in_error in self.active_requests:
                    request_in_error = self.active_requests.pop(req_id_in_error)
                    try:
                        request_in_error.callback([], True) # Notify with is_done=True due to error
                    except Exception as cb_error:
                        logger.error(f"Error in error callback for {request_in_error.request_id}: {str(cb_error)}")
                    self.request_to_batch.pop(request_in_error.request_id, None)
                    self.stats["active_requests"] -= 1
            
            # Clean up the problematic batch
            self.active_batches.pop(batch_id, None)