    """
    Dynamic batching implementation for efficient inference.
    """

    import logging
    import threading
    import time
    import queue
    from typing import Dict, Any, Optional, List, Callable, Union, Tuple

    import numpy as np

    logger = logging.getLogger(__name__)

    class BatchRequest:
        """Class representing a request in the batch queue."""
    
        def __init__(self, inputs, callback, request_id=None):
            """
            Initialize a batch request.
        
            Args:
                inputs: Input data for the request
                callback: Function to call with the result
                request_id: Optional request identifier
            """
            self.inputs = inputs
            self.callback = callback
            self.request_id = request_id or f"req_{time.time()}_{id(self)}"
            self.submit_time = time.time()
            self.processing_start_time = None
            self.processing_end_time = None


    class DynamicBatcher:
        """
        Dynamic batching implementation for efficient inference.
    
        Accumulates inference requests up to a maximum batch size
        or until a timeout is reached, then processes the batch at once.
        """
    
        def __init__(self, 
                    compute_fn: Callable[[Any], Any],
                    max_batch_size: int = 32,
                    max_wait_time_ms: int = 100,
                    min_batch_size: int = 1,
                    extra_wait_factor: float = 0.5,
                    performance_tracker: Optional[Any] = None):
            """
            Initialize the dynamic batcher.
        
            Args:
                compute_fn: Function to call with batched inputs
                max_batch_size: Maximum batch size
                max_wait_time_ms: Maximum time to wait for a batch to fill (ms)
                min_batch_size: Minimum batch size to process
                extra_wait_factor: Factor to wait longer if batch size is growing
                performance_tracker: Performance tracker for metrics
            """
            self.compute_fn = compute_fn
            self.max_batch_size = max_batch_size
            self.max_wait_time_ms = max_wait_time_ms
            self.min_batch_size = min_batch_size
            self.extra_wait_factor = extra_wait_factor
            self.performance_tracker = performance_tracker
        
            # Request queue and processing thread
            self.request_queue = queue.Queue()
            self.worker_thread = None
            self.stop_event = threading.Event()
        
            # Lock for thread-safe operations
            self.lock = threading.RLock()
        
            # Statistics
            self.stats = {
                "total_batches": 0,
                "total_requests": 0,
                "avg_batch_size": 0,
                "avg_wait_time_ms": 0,
                "avg_processing_time_ms": 0,
                "request_throughput": 0,
                "batch_throughput": 0,
                "queue_time_p50_ms": 0,
                "queue_time_p95_ms": 0,
                "queue_time_p99_ms": 0,
            }
        
            # Recent stats for moving averages
            self.recent_batch_sizes = []
            self.recent_wait_times = []
            self.recent_processing_times = []
            self.recent_queue_times = []
        
            # Start the worker thread
            self.start()
    
        def start(self):
            """Start the batcher worker thread."""
            with self.lock:
                if self.worker_thread is None or not self.worker_thread.is_alive():
                    self.stop_event.clear()
                    self.worker_thread = threading.Thread(
                        target=self._process_batches,
                        name="DynamicBatcher-Worker"
                    )
                    self.worker_thread.daemon = True
                    self.worker_thread.start()
                    logger.info("Dynamic batcher worker thread started")
    
        def stop(self):
            """Stop the batcher worker thread."""
            with self.lock:
                if self.worker_thread and self.worker_thread.is_alive():
                    self.stop_event.set()
                    self.worker_thread.join(timeout=5.0)
                    logger.info("Dynamic batcher worker thread stopped")
    
        def submit(self, 
                  inputs: Any, 
                  callback: Optional[Callable[[Any], None]] = None) -> str:
            """
            Submit a request to the batcher.
        
            Args:
                inputs: Input data to process
                callback: Function to call with the result
            
            Returns:
                request_id: ID for the submitted request
            """
            # Create a request
            request = BatchRequest(inputs, callback)
        
            # Add to queue
            self.request_queue.put(request)
        
            # Update stats
            with self.lock:
                self.stats["total_requests"] += 1
        
            return request.request_id
    
        def process_sync(self, inputs: Any) -> Any:
            """
            Process an input synchronously.
        
            This is a blocking call that waits for the result.
        
            Args:
                inputs: Input data to process
            
            Returns:
                result: Processed output
            """
            result_event = threading.Event()
            result_container = {"result": None}
        
            def callback(output):
                result_container["result"] = output
                result_event.set()
        
            # Submit the request
            self.submit(inputs, callback)
        
            # Wait for the result
            result_event.wait()
        
            return result_container["result"]
    
        def _process_batches(self):
            """Main worker loop to process batches of requests."""
            try:
                while not self.stop_event.is_set():
                    try:
                        batch, start_wait_time = self._collect_batch()
                    
                        if not batch:
                            # No requests, sleep briefly
                            time.sleep(0.005)
                            continue
                    
                        # Process the batch
                        batch_size = len(batch)
                    
                        # Update statistics for wait time
                        total_wait_time_ms = (time.time() - start_wait_time) * 1000
                        with self.lock:
                            self.recent_wait_times.append(total_wait_time_ms)
                            if len(self.recent_wait_times) > 100:
                                self.recent_wait_times.pop(0)
                            self.stats["avg_wait_time_ms"] = sum(self.recent_wait_times) / len(self.recent_wait_times)
                    
                        # Prepare batch inputs - handle different types of inputs
                        try:
                            batch_inputs = self._collate_batch([req.inputs for req in batch])
                        except Exception as e:
                            logger.error(f"Error collating batch: {str(e)}")
                            # Process individually as fallback
                            for req in batch:
                                try:
                                    req.processing_start_time = time.time()
                                    result = self.compute_fn(req.inputs)
                                    req.processing_end_time = time.time()
                                    if req.callback:
                                        req.callback(result)
                                except Exception as e:
                                    logger.error(f"Error processing individual request: {str(e)}")
                            continue
                    
                        # Process the batch
                        batch_start_time = time.time()
                        for req in batch:
                            req.processing_start_time = batch_start_time
                    
                        try:
                            batch_results = self.compute_fn(batch_inputs)
                            batch_end_time = time.time()
                        
                            # Update processing time stats
                            processing_time_ms = (batch_end_time - batch_start_time) * 1000
                            with self.lock:
                                self.recent_processing_times.append(processing_time_ms)
                                if len(self.recent_processing_times) > 100:
                                    self.recent_processing_times.pop(0)
                                self.stats["avg_processing_time_ms"] = sum(self.recent_processing_times) / len(self.recent_processing_times)
                        
                            # Update batch size stats
                            with self.lock:
                                self.recent_batch_sizes.append(batch_size)
                                if len(self.recent_batch_sizes) > 100:
                                    self.recent_batch_sizes.pop(0)
                                self.stats["avg_batch_size"] = sum(self.recent_batch_sizes) / len(self.recent_batch_sizes)
                                self.stats["total_batches"] += 1
                        
                            # Calculate queue times
                            queue_times = []
                            for req in batch:
                                req.processing_end_time = batch_end_time
                                queue_time = (req.processing_start_time - req.submit_time) * 1000
                                queue_times.append(queue_time)
                        
                            with self.lock:
                                self.recent_queue_times.extend(queue_times)
                                if len(self.recent_queue_times) > 1000:
                                    self.recent_queue_times = self.recent_queue_times[-1000:]
                            
                                # Calculate percentiles
                                if self.recent_queue_times:
                                    self.recent_queue_times.sort()
                                    length = len(self.recent_queue_times)
                                    self.stats["queue_time_p50_ms"] = self.recent_queue_times[int(length * 0.5)]
                                    self.stats["queue_time_p95_ms"] = self.recent_queue_times[int(length * 0.95)]
                                    self.stats["queue_time_p99_ms"] = self.recent_queue_times[int(length * 0.99)]
                        
                            # Deliver results to callbacks
                            for i, req in enumerate(batch):
                                try:
                                    if req.callback:
                                        if isinstance(batch_results, list) and len(batch_results) == len(batch):
                                            req.callback(batch_results[i])
                                        else:
                                            # Handle case where compute_fn doesn't return a list
                                            # or the list doesn't match batch length
                                            req.callback(batch_results)
                                except Exception as e:
                                    logger.error(f"Error in callback for request {req.request_id}: {str(e)}")
                    
                        except Exception as e:
                            logger.error(f"Error processing batch: {str(e)}")
                            # Notify all callbacks of failure
                            for req in batch:
                                try:
                                    if req.callback:
                                        req.callback(None)  # Indicate failure
                                except Exception as cb_err:
                                    logger.error(f"Error in failure callback: {str(cb_err)}")
                
                    except Exception as e:
                        logger.error(f"Error in batch processing loop: {str(e)}")
                        time.sleep(0.1)  # Brief pause to prevent tight error loops
        
        except Exception as e:
            logger.error(f"Fatal error in dynamic batcher worker: {str(e)}")
            # Try to restart the worker
            time.sleep(1.0)
            if not self.stop_event.is_set():
                self.start()
    
    def _collect_batch(self) -> Tuple[List[BatchRequest], float]:
        """
        Collect requests into a batch, waiting up to max_wait_time_ms.
        
        Returns:
            (batch, start_time): List of BatchRequest objects and batch start time
        """
        start_time = time.time()
        batch = []
        max_wait_sec = self.max_wait_time_ms / 1000.0
        
        # Get first request or return empty batch if queue is empty
        try:
            # Wait for the first request
            first_request = self.request_queue.get(timeout=0.005)  # 5ms timeout for responsiveness
            batch.append(first_request)
        except queue.Empty:
            return [], start_time
        
        # Now try to fill the batch up to max_batch_size or until timeout
        deadline = start_time + max_wait_sec
        
        # Dynamic timeout based on batch growth
        last_batch_change_time = time.time()
        
        while len(batch) < self.max_batch_size:
            # Calculate remaining time
            now = time.time()
            remaining_time = deadline - now
            
            # If batch size increased recently, extend the deadline a bit
            if now - last_batch_change_time < max_wait_sec * self.extra_wait_factor:
                # Extend deadline a bit if the batch is still actively growing
                remaining_time = max(remaining_time, max_wait_sec * self.extra_wait_factor)
            
            # If we've reached minimum batch size and timeout, break
            if len(batch) >= self.min_batch_size and remaining_time <= 0:
                break
            
            # Try to get a request
            try:
                # Use a small timeout to remain responsive
                request = self.request_queue.get(timeout=min(0.005, max(0, remaining_time)))
                batch.append(request)
                last_batch_change_time = time.time()  # Update last change time
            except queue.Empty:
                # If we've waited long enough or reached min batch size, break
                if remaining_time <= 0 and len(batch) >= self.min_batch_size:
                    break
                
                # Small sleep to avoid tight loop
                if remaining_time > 0.01:  # Only sleep if we have significant time remaining
                    time.sleep(0.001)
        
        return batch, start_time
    
    def _collate_batch(self, batch_inputs: List[Any]) -> Any:
        """
        Collate a list of inputs into a batched input for the compute function.
        
        This method handles different types of inputs (torch tensors, numpy arrays, etc.)
        and attempts to batch them appropriately.
        
        Args:
            batch_inputs: List of input data
            
        Returns:
            batched_input: Batched input compatible with the compute function
        """
        if not batch_inputs:
            return None
        
        # Try to determine the type of the first input
        sample = batch_inputs[0]
        
        try:
            # Handle numpy arrays
            if hasattr(sample, 'dtype') and hasattr(sample, 'shape') and hasattr(sample, '__array__'):
                # Likely a numpy array or similar
                import numpy as np
                return np.stack(batch_inputs, axis=0)
            
            # Handle PyTorch tensors
            elif hasattr(sample, 'shape') and hasattr(sample, 'dtype') and hasattr(sample, 'to'):
                # Likely a PyTorch tensor
                import torch
                return torch.stack(batch_inputs, dim=0)
            
            # Handle dictionaries (e.g., HuggingFace model inputs)
            elif isinstance(sample, dict):
                # For each key, stack all values
                result = {}
                for key in sample:
                    # Skip keys not present in all samples
                    if all(key in x for x in batch_inputs):
                        values = [x[key] for x in batch_inputs]
                        
                        # Handle different value types
                        if isinstance(values[0], (int, float, bool, str)):
                            # Simple types - just make a list
                            result[key] = values
                        elif hasattr(values[0], 'shape') and hasattr(values[0], 'dtype'):
                            # Tensors or arrays - try stacking
                            try:
                                if hasattr(values[0], 'to'):  # PyTorch tensor
                                    import torch
                                    result[key] = torch.stack(values, dim=0)
                                else:  # Numpy array or similar
                                    import numpy as np
                                    result[key] = np.stack(values, axis=0)
                            except Exception as e:
                                logger.warning(f"Failed to stack values for key {key}: {e}")
                                result[key] = values
                        else:
                            # Just use the list
                            result[key] = values
                
                return result
            
            # Handle lists/tuples
            elif isinstance(sample, (list, tuple)):
                # For simple lists of the same length, try to collate element-wise
                # Each position in the list becomes a batch
                if all(isinstance(x, (list, tuple)) and len(x) == len(sample) for x in batch_inputs):
                    result = []
                    for i in range(len(sample)):
                        # Extract ith element from each input
                        elements = [x[i] for x in batch_inputs]
                        
                        # Recursively collate if needed
                        if isinstance(elements[0], (dict, list, tuple)) or hasattr(elements[0], 'shape'):
                            result.append(self._collate_batch(elements))
                        else:
                            result.append(elements)
                    return result
                else:
                    # If elements are not compatible, just return the list
                    return batch_inputs
            
            # Fall back to returning the list of inputs
            else:
                return batch_inputs
                
        except Exception as e:
            logger.warning(f"Error collating batch: {str(e)}. Falling back to simple list.")
            return batch_inputs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the dynamic batcher."""
        with self.lock:
            stats = dict(self.stats)  # Return a copy
            
            # Calculate throughput based on recent processing times
            if self.recent_processing_times:
                avg_batch_time = sum(self.recent_processing_times) / len(self.recent_processing_times) / 1000.0  # Convert to seconds
                avg_batch_size = sum(self.recent_batch_sizes) / len(self.recent_batch_sizes) if self.recent_batch_sizes else 0
                
                if avg_batch_time > 0:
                    stats["batch_throughput"] = 1.0 / avg_batch_time  # batches/second
                    stats["request_throughput"] = avg_batch_size / avg_batch_time  # requests/second
            
            # Add queue size
            stats["current_queue_size"] = self.request_queue.qsize()
            
            return stats