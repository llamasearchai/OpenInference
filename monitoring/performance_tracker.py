import time
import threading
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import json
import os
from collections import deque
import uuid

import numpy as np
import psutil

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class PerformanceTracker:
    """
    Performance tracking system for model inference.
    
    Tracks and analyzes latency, throughput, memory usage, and other
    performance metrics during inference.
    """
    
    def __init__(self, 
                 window_size: int = 1000,
                 log_interval: float = 60.0,
                 record_gpu_metrics: bool = True,
                 export_metrics: bool = True):
        """
        Initialize the performance tracker.
        
        Args:
            window_size: Number of requests to keep in sliding window
            log_interval: Interval in seconds to log performance metrics
            record_gpu_metrics: Whether to record GPU metrics
            export_metrics: Whether to export metrics to file
        """
        self.window_size = window_size
        self.log_interval = log_interval
        self.record_gpu_metrics = record_gpu_metrics and TORCH_AVAILABLE
        self.export_metrics = export_metrics
        
        # Sliding window of request metrics
        self.request_metrics = deque(maxlen=window_size)
        
        # System metrics history
        self.system_metrics = []
        
        # Global statistics
        self.global_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_inference_ms": 0,
            "start_time": time.time()
        }
        
        # Monitoring thread
        self.monitor_thread = None
        self.is_running = False
        
        # Metrics export
        self.metrics_dir = "metrics"
        if self.export_metrics:
            os.makedirs(self.metrics_dir, exist_ok=True)
    
    def start(self):
        """Start the performance monitor."""
        if self.is_running:
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitor started")
    
    def stop(self):
        """Stop the performance monitor."""
        self.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
            
        logger.info("Performance monitor stopped")
        
        # Export final metrics
        if self.export_metrics:
            self.export_metrics_to_file()
    
    def start_request(self, request_id: str, model_name: str) -> Dict[str, Any]:
        """
        Start tracking a new request.
        
        Args:
            request_id: Unique request identifier
            model_name: Name of the model
            
        Returns:
            tracking_info: Information to pass to finish_request
        """
        return {
            "request_id": request_id,
            "model_name": model_name,
            "start_time": time.time()
        }
    
    def finish_request(self, 
                      tracking_info: Dict[str, Any],
                      batch_size: int = 1,
                      input_shape: Optional[List[int]] = None,
                      output_shape: Optional[List[int]] = None,
                      compute_start_time: Optional[float] = None,
                      success: bool = True):
        """
        Finish tracking a request.
        
        Args:
            tracking_info: Information from start_request
            batch_size: Batch size for this request
            input_shape: Shape of the input tensor
            output_shape: Shape of the output tensor
            compute_start_time: When actual computation started (if different from start_time)
            success: Whether the request was successful
        """
        end_time = time.time()
        request_id = tracking_info.get("request_id", "unknown")
        model_name = tracking_info.get("model_name", "unknown")
        start_time = tracking_info.get("start_time", end_time)
        
        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000
        
        # Calculate compute time if provided
        if compute_start_time is not None:
            queue_time_ms = (compute_start_time - start_time) * 1000
            compute_time_ms = (end_time - compute_start_time) * 1000
        else:
            queue_time_ms = 0
            compute_time_ms = total_time_ms
        
        # Record token count if output_shape is provided (for LLMs)
        token_count = 0
        if output_shape is not None and len(output_shape) > 0:
            # Assume last dimension is sequence length for LLMs
            token_count = output_shape[-1]
        
        # Create metrics record
        metrics = {
            "request_id": request_id,
            "model_name": model_name,
            "batch_size": batch_size,
            "start_time": start_time,
            "end_time": end_time,
            "total_time_ms": total_time_ms,
            "queue_time_ms": queue_time_ms,
            "compute_time_ms": compute_time_ms,
            "token_count": token_count,
            "success": success,
            "input_shape": input_shape,
            "output_shape": output_shape
        }
        
        # Add to sliding window
        self.request_metrics.append(metrics)
        
        # Update global stats
        self.global_stats["total_requests"] += 1
        if success:
            self.global_stats["successful_requests"] += 1
        else:
            self.global_stats["failed_requests"] += 1
        
        self.global_stats["total_tokens"] += token_count
        self.global_stats["total_inference_ms"] += compute_time_ms
        
        # Log slow requests
        if total_time_ms > 1000:  # Slower than 1 second
            logger.warning(f"Slow request: {request_id}, model: {model_name}, time: {total_time_ms:.2f}ms")
        
        return metrics
    
    def _monitor_loop(self):
        """Background thread to monitor system metrics."""
        last_log_time = time.time()
        
        while self.is_running:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Limit history to window_size
                if len(self.system_metrics) > self.window_size:
                    self.system_metrics = self.system_metrics[-self.window_size:]
                
                # Log metrics at interval
                current_time = time.time()
                if current_time - last_log_time > self.log_interval:
                    self._log_performance_summary()
                    last_log_time = current_time
                
                # Sleep for a bit
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {str(e)}")
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3)
        }
        
        # Add GPU metrics if enabled
        if self.record_gpu_metrics and TORCH_AVAILABLE:
            try:
                gpu_metrics = {}
                for i in range(torch.cuda.device_count()):
                    gpu_metrics[f"gpu_{i}_util_percent"] = torch.cuda.utilization(i)
                    gpu_metrics[f"gpu_{i}_memory_used_gb"] = torch.cuda.memory_allocated(i) / (1024**3)
                    gpu_metrics[f"gpu_{i}_memory_total_gb"] = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                metrics.update(gpu_metrics)
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {str(e)}")
        
        return metrics
    
    def _log_performance_summary(self):
        """Log a summary of recent performance metrics."""
        if not self.request_metrics:
            return
        
        # Calculate summary statistics
        latencies = [m["total_time_ms"] for m in self.request_metrics]
        compute_times = [m["compute_time_ms"] for m in self.request_metrics]
        queue_times = [m["queue_time_ms"] for m in self.request_metrics]
        token_counts = [m["token_count"] for m in self.request_metrics]
        success_rate = sum(1 for m in self.request_metrics if m["success"]) / len(self.request_metrics) * 100
        
        # Calculate percentiles
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Calculate throughput (requests per second)
        if self.request_metrics:
            window_start = self.request_metrics[0]["start_time"]
            window_end = self.request_metrics[-1]["end_time"]
            window_duration = max(1e-6, window_end - window_start)
            requests_per_second = len(self.request_metrics) / window_duration
            
            # Token throughput for LLMs
            token_throughput = sum(token_counts) / window_duration
        else:
            requests_per_second = 0
            token_throughput = 0
        
        # System metrics
        if self.system_metrics:
            avg_cpu = np.mean([m["cpu_percent"] for m in self.system_metrics])
            avg_memory = np.mean([m["memory_percent"] for m in self.system_metrics])
        else:
            avg_cpu = 0
            avg_memory = 0
        
        # Log summary
        logger.info(f"Performance summary (last {len(self.request_metrics)} requests):")
        logger.info(f"  Throughput: {requests_per_second:.2f} req/s, {token_throughput:.2f} tokens/s")
        logger.info(f"  Latency (ms): p50={p50_latency:.2f}, p95={p95_latency:.2f}, p99={p99_latency:.2f}")
        logger.info(f"  Compute time (ms): avg={np.mean(compute_times):.2f}, max={max(compute_times):.2f}")
        logger.info(f"  Queue time (ms): avg={np.mean(queue_times):.2f}, max={max(queue_times):.2f}")
        logger.info(f"  Success rate: {success_rate:.2f}%")
        logger.info(f"  System: CPU={avg_cpu:.2f}%, Memory={avg_memory:.2f}%")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self.request_metrics:
            return {
                "request_count": 0,
                "latency_ms": {"p50": 0, "p95": 0, "p99": 0, "avg": 0, "min": 0, "max": 0},
                "throughput": {"requests_per_second": 0, "tokens_per_second": 0},
                "success_rate": 0,
                "system": {}
            }
        
        # Calculate summary statistics
        latencies = [m["total_time_ms"] for m in self.request_metrics]
        compute_times = [m["compute_time_ms"] for m in self.request_metrics]
        queue_times = [m["queue_time_ms"] for m in self.request_metrics]
        token_counts = [m["token_count"] for m in self.request_metrics]
        
        # Success rate
        success_rate = sum(1 for m in self.request_metrics if m["success"]) / len(self.request_metrics) * 100
        
        # Calculate throughput
        if self.request_metrics:
            window_start = self.request_metrics[0]["start_time"]
            window_end = self.request_metrics[-1]["end_time"]
            window_duration = max(1e-6, window_end - window_start)
            requests_per_second = len(self.request_metrics) / window_duration
            token_throughput = sum(token_counts) / window_duration
        else:
            requests_per_second = 0
            token_throughput = 0
        
        # System metrics
        system = {}
        if self.system_metrics:
            system = {
                "cpu_percent": {
                    "avg": np.mean([m["cpu_percent"] for m in self.system_metrics]),
                    "max": max([m["cpu_percent"] for m in self.system_metrics])
                },
                "memory_percent": {
                    "avg": np.mean([m["memory_percent"] for m in self.system_metrics]),
                    "max": max([m["memory_percent"] for m in self.system_metrics])
                }
            }
            
            # Add GPU metrics if available
            if self.record_gpu_metrics and TORCH_AVAILABLE:
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    util_key = f"gpu_{i}_util_percent"
                    mem_key = f"gpu_{i}_memory_used_gb"
                    
                    if all(util_key in m for m in self.system_metrics):
                        system[util_key] = {
                            "avg": np.mean([m[util_key] for m in self.system_metrics]),
                            "max": max([m[util_key] for m in self.system_metrics])
                        }
                    
                    if all(mem_key in m for m in self.system_metrics):
                        system[mem_key] = {
                            "avg": np.mean([m[mem_key] for m in self.system_metrics]),
                            "max": max([m[mem_key] for m in self.system_metrics])
                        }
        
        # Global statistics
        uptime = time.time() - self.global_stats["start_time"]
        global_metrics = {
            "total_requests": self.global_stats["total_requests"],
            "successful_requests": self.global_stats["successful_requests"],
            "failed_requests": self.global_stats["failed_requests"],
            "total_tokens": self.global_stats["total_tokens"],
            "uptime_seconds": uptime,
            "global_throughput": {
                "requests_per_second": self.global_stats["total_requests"] / max(1, uptime),
                "tokens_per_second": self.global_stats["total_tokens"] / max(1, uptime)
            }
        }
        
        # Build complete metrics summary
        return {
            "request_count": len(self.request_metrics),
            "latency_ms": {
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "avg": np.mean(latencies),
                "min": min(latencies),
                "max": max(latencies)
            },
            "compute_time_ms": {
                "avg": np.mean(compute_times),
                "max": max(compute_times)
            },
            "queue_time_ms": {
                "avg": np.mean(queue_times),
                "max": max(queue_times)
            },
            "throughput": {
                "requests_per_second": requests_per_second,
                "tokens_per_second": token_throughput
            },
            "success_rate": success_rate,
            "system": system,
            "global": global_metrics
        }
    
    def export_metrics_to_file(self):
        """Export metrics to JSON file."""
        if not self.export_metrics:
            return
        
        try:
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"metrics_{timestamp}_{uuid.uuid4().hex[:8]}.json"
            filepath = os.path.join(self.metrics_dir, filename)
            
            # Export data
            data = {
                "summary": self.get_metrics_summary(),
                "request_metrics": list(self.request_metrics),
                "system_metrics": self.system_metrics
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Exported metrics to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            return None