"""
Metrics collection and monitoring for inference performance.
"""

import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class InferenceMetric:
    """Single inference metric measurement."""
    name: str
    value: Union[float, int]
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels
        }

class MetricsCollector:
    """
    Collects and aggregates metrics for inference performance monitoring.
    
    Provides methods for recording latency, throughput, memory usage,
    and other key performance indicators.
    """
    
    def __init__(self, 
                export_path: Optional[str] = None,
                export_interval: int = 60,
                max_metrics_history: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            export_path: Path to export metrics (None to disable)
            export_interval: Interval in seconds to export metrics
            max_metrics_history: Maximum number of metrics to keep in memory
        """
        self.metrics = {}
        self.metrics_history = {}
        self.current_requests = {}
        self.export_path = export_path
        self.export_interval = export_interval
        self.max_metrics_history = max_metrics_history
        
        # For thread safety
        self._lock = threading.Lock()
        self._export_queue = queue.Queue()
        
        # Start exporter thread if needed
        self._exporter_thread = None
        if export_path:
            os.makedirs(export_path, exist_ok=True)
            self._exporter_thread = threading.Thread(
                target=self._metrics_exporter,
                daemon=True
            )
            self._exporter_thread.start()
            logger.info(f"Started metrics exporter thread (interval: {export_interval}s)")
        
        # Initialize basic metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize basic metrics with default values."""
        with self._lock:
            # System metrics
            self.metrics["system.memory.used_mb"] = 0
            self.metrics["system.memory.total_mb"] = 0
            self.metrics["system.cpu.utilization"] = 0
            self.metrics["system.gpu.utilization"] = 0
            self.metrics["system.gpu.memory.used_mb"] = 0
            self.metrics["system.gpu.memory.total_mb"] = 0
            
            # Inference metrics
            self.metrics["inference.requests.active"] = 0
            self.metrics["inference.requests.completed"] = 0
            self.metrics["inference.requests.failed"] = 0
            self.metrics["inference.latency.p50_ms"] = 0
            self.metrics["inference.latency.p90_ms"] = 0
            self.metrics["inference.latency.p99_ms"] = 0
            self.metrics["inference.throughput.tokens_per_second"] = 0
            self.metrics["inference.throughput.requests_per_second"] = 0
            
            # Batching metrics
            self.metrics["batching.batch_size.avg"] = 0
            self.metrics["batching.queue_depth"] = 0
            self.metrics["batching.prefill.tokens_per_second"] = 0
            self.metrics["batching.decode.tokens_per_second"] = 0
            
            # Initialize history for each metric
            for name in self.metrics:
                self.metrics_history[name] = []
    
    def record_metric(self, name: str, value: Union[float, int], labels: Optional[Dict[str, str]] = None):
        """
        Record a new metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels to attach to metric
        """
        with self._lock:
            # Update current value
            self.metrics[name] = value
            
            # Create metric object
            metric = InferenceMetric(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            )
            
            # Add to history with limit on size
            history = self.metrics_history.get(name, [])
            history.append(metric)
            if len(history) > self.max_metrics_history:
                history = history[-self.max_metrics_history:]
            self.metrics_history[name] = history
            
            # Add to export queue if exporting is enabled
            if self.export_path:
                self._export_queue.put(metric)
    
    def start_request(self, request_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Start tracking a new inference request.
        
        Args:
            request_id: Unique ID for the request
            metadata: Optional request metadata
        """
        with self._lock:
            self.current_requests[request_id] = {
                "start_time": time.time(),
                "first_token_time": None,
                "end_time": None,
                "tokens_generated": 0,
                "is_completed": False,
                "metadata": metadata or {}
            }
            
            # Update active requests count
            self.metrics["inference.requests.active"] = len(self.current_requests)
    
    def record_first_token(self, request_id: str):
        """
        Record when first token is generated for a request.
        
        Args:
            request_id: Request ID
        """
        with self._lock:
            if request_id in self.current_requests:
                self.current_requests[request_id]["first_token_time"] = time.time()
    
    def add_generated_tokens(self, request_id: str, token_count: int):
        """
        Add to token count for a request.
        
        Args:
            request_id: Request ID
            token_count: Number of tokens to add
        """
        with self._lock:
            if request_id in self.current_requests:
                self.current_requests[request_id]["tokens_generated"] += token_count
    
    def end_request(self, request_id: str, success: bool = True):
        """
        End tracking for a request and record final metrics.
        
        Args:
            request_id: Request ID
            success: Whether request completed successfully
        """
        with self._lock:
            if request_id not in self.current_requests:
                return
            
            # Record end time
            request = self.current_requests[request_id]
            request["end_time"] = time.time()
            request["is_completed"] = True
            
            # Calculate latency
            total_latency_ms = (request["end_time"] - request["start_time"]) * 1000
            
            # Calculate time to first token if available
            ttft_ms = None
            if request["first_token_time"] is not None:
                ttft_ms = (request["first_token_time"] - request["start_time"]) * 1000
            
            # Record latency metrics
            self._record_latency(total_latency_ms)
            
            # Update throughput metrics
            tokens_generated = request["tokens_generated"]
            if tokens_generated > 0 and total_latency_ms > 0:
                tokens_per_second = tokens_generated / (total_latency_ms / 1000)
                self._update_throughput(tokens_per_second)
            
            # Update request counts
            if success:
                self.metrics["inference.requests.completed"] += 1
            else:
                self.metrics["inference.requests.failed"] += 1
            
            # Remove from active requests
            del self.current_requests[request_id]
            self.metrics["inference.requests.active"] = len(self.current_requests)
    
    def _record_latency(self, latency_ms: float):
        """
        Record request latency and update percentiles.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        # Add to latency history
        history_key = "inference.latency.history"
        latencies = self.metrics_history.get(history_key, [])
        
        metric = InferenceMetric(
            name=history_key,
            value=latency_ms,
            timestamp=time.time()
        )
        
        latencies.append(metric)
        if len(latencies) > self.max_metrics_history:
            latencies = latencies[-self.max_metrics_history:]
        self.metrics_history[history_key] = latencies
        
        # Update percentiles if we have enough data
        if len(latencies) >= 10:
            values = sorted([m.value for m in latencies])
            p50_idx = int(len(values) * 0.5)
            p90_idx = int(len(values) * 0.9)
            p99_idx = int(len(values) * 0.99)
            
            self.metrics["inference.latency.p50_ms"] = values[p50_idx]
            self.metrics["inference.latency.p90_ms"] = values[p90_idx]
            self.metrics["inference.latency.p99_ms"] = values[p99_idx]
    
    def _update_throughput(self, tokens_per_second: float):
        """
        Update throughput metrics using exponential moving average.
        
        Args:
            tokens_per_second: Tokens per second for current request
        """
        current = self.metrics["inference.throughput.tokens_per_second"]
        if current == 0:
            # Initialize with first value
            self.metrics["inference.throughput.tokens_per_second"] = tokens_per_second
        else:
            # Apply EMA with 0.05 weight for new sample
            self.metrics["inference.throughput.tokens_per_second"] = (
                current * 0.95 + tokens_per_second * 0.05
            )
    
    def update_system_metrics(self, metrics: Dict[str, Union[float, int]]):
        """
        Update system metrics (CPU, GPU, memory, etc.).
        
        Args:
            metrics: Dictionary of system metrics to update
        """
        with self._lock:
            for name, value in metrics.items():
                if name in self.metrics:
                    self.record_metric(name, value)
    
    def update_batch_metrics(self, metrics: Dict[str, Union[float, int]]):
        """
        Update batching-related metrics.
        
        Args:
            metrics: Dictionary of batching metrics to update
        """
        with self._lock:
            for name, value in metrics.items():
                if name.startswith("batching."):
                    self.record_metric(name, value)
    
    def get_current_metrics(self) -> Dict[str, Union[float, int]]:
        """Get all current metric values."""
        with self._lock:
            return dict(self.metrics)
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get history for a specific metric.
        
        Args:
            name: Metric name
            limit: Maximum number of data points to return
            
        Returns:
            List of metric data points
        """
        with self._lock:
            history = self.metrics_history.get(name, [])
            return [m.as_dict() for m in history[-limit:]]
    
    def _metrics_exporter(self):
        """Background thread for exporting metrics to disk."""
        last_export_time = time.time()
        
        while True:
            try:
                # Check if it's time for a full export
                current_time = time.time()
                if current_time - last_export_time >= self.export_interval:
                    self._export_all_metrics()
                    last_export_time = current_time
                
                # Process individual metrics from queue
                try:
                    metric = self._export_queue.get(timeout=1.0)
                    self._export_metric(metric)
                    self._export_queue.task_done()
                except queue.Empty:
                    # No metrics to export
                    pass
                
            except Exception as e:
                logger.error(f"Error in metrics exporter: {str(e)}", exc_info=True)
                time.sleep(5)  # Avoid tight loop on error
    
    def _export_all_metrics(self):
        """Export all current metrics to disk."""
        if not self.export_path:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.export_path, f"metrics_snapshot_{timestamp}.json")
            
            with open(filename, "w") as f:
                json.dump({
                    "timestamp": time.time(),
                    "metrics": self.get_current_metrics()
                }, f, indent=2)
            
            logger.debug(f"Exported metrics snapshot to {filename}")
        except Exception as e:
            logger.error(f"Failed to export metrics snapshot: {str(e)}")
    
    def _export_metric(self, metric: InferenceMetric):
        """Export a single metric to disk."""
        if not self.export_path:
            return
        
        try:
            # Append to daily log file
            date_str = datetime.fromtimestamp(metric.timestamp).strftime("%Y%m%d")
            filename = os.path.join(self.export_path, f"metrics_{date_str}.jsonl")
            
            with open(filename, "a") as f:
                f.write(json.dumps(metric.as_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to export metric {metric.name}: {str(e)}")