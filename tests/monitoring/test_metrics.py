import unittest
import time
import os
import shutil
from unittest.mock import patch, MagicMock

from monitoring.metrics import MetricsCollector, InferenceMetric

class TestMetricsCollector(unittest.TestCase):

    def setUp(self):
        self.export_path = "test_metrics_export"
        os.makedirs(self.export_path, exist_ok=True)
        self.collector = MetricsCollector(export_path=self.export_path, export_interval=1)

    def tearDown(self):
        if self.collector._exporter_thread and self.collector._exporter_thread.is_alive():
            # Signal exporter to stop (implementation-dependent, here we just join with timeout)
            # In a real scenario, you might need a more graceful shutdown mechanism for the thread
            self.collector._export_queue.put(None) # Assuming None can signal shutdown or use a dedicated sentinel
            self.collector._exporter_thread.join(timeout=2)
        
        # Clean up created directories and files if any
        if os.path.exists(self.export_path):
            shutil.rmtree(self.export_path)

    def test_initialization(self):
        self.assertIsNotNone(self.collector)
        self.assertEqual(self.collector.export_path, self.export_path)
        self.assertIn("system.memory.used_mb", self.collector.metrics)
        self.assertTrue(self.collector._exporter_thread.is_alive())

    def test_record_metric(self):
        self.collector.record_metric("test.metric", 10.5, labels={"env": "test"})
        self.assertEqual(self.collector.metrics["test.metric"], 10.5)
        self.assertEqual(len(self.collector.metrics_history["test.metric"]), 1)
        self.assertEqual(self.collector.metrics_history["test.metric"][0].value, 10.5)
        self.assertEqual(self.collector.metrics_history["test.metric"][0].labels, {"env": "test"})
        
        # Check if metric is added to export queue
        # We expect one metric due to this test and potentially others from initialization
        # For simplicity in this test, let's clear the queue first or check specific metric
        
        # To ensure no interference from other tests or initial metrics, let's try to get our specific metric
        # This might be tricky if other metrics are also being put on the queue by the init process.
        # A more robust way would be to mock the queue or have a way to inspect its contents more directly.

        # Drain the queue to remove any initial metrics (if any are put by __init__ which they are not for export_queue)
        while not self.collector._export_queue.empty():
            self.collector._export_queue.get_nowait()

        self.collector.record_metric("test.metric.for.queue", 1.0)
        exported_metric = self.collector._export_queue.get_nowait()
        self.assertEqual(exported_metric.name, "test.metric.for.queue")
        self.assertEqual(exported_metric.value, 1.0)

    def test_record_metric_history_limit(self):
        self.collector.max_metrics_history = 2 # Set a small limit for testing
        self.collector.record_metric("test.history.limit", 1)
        self.collector.record_metric("test.history.limit", 2)
        self.collector.record_metric("test.history.limit", 3)
        self.assertEqual(len(self.collector.metrics_history["test.history.limit"]), 2)
        self.assertEqual(self.collector.metrics_history["test.history.limit"][0].value, 2)
        self.assertEqual(self.collector.metrics_history["test.history.limit"][1].value, 3)

    def test_request_tracking(self):
        request_id = "test_req_123"
        self.collector.start_request(request_id, metadata={"model": "test_model"})
        self.assertIn(request_id, self.collector.current_requests)
        self.assertEqual(self.collector.metrics["inference.requests.active"], 1)

        self.collector.record_first_token(request_id)
        self.assertIsNotNone(self.collector.current_requests[request_id]["first_token_time"])

        self.collector.add_generated_tokens(request_id, 100)
        self.assertEqual(self.collector.current_requests[request_id]["tokens_generated"], 100)

        # Simulate some time passing for latency calculation
        # We need to mock time.time() for predictable latency testing
        start_time = self.collector.current_requests[request_id]["start_time"]
        first_token_time = self.collector.current_requests[request_id]["first_token_time"]
        
        # Mock time.time() to control end_time for predictable latency
        # Ensure that the mocked end_time is after start_time and first_token_time
        # Add a small delay to ensure first_token_time is indeed before end_time if it exists
        mock_end_time = first_token_time + 0.1 if first_token_time else start_time + 0.2
        if mock_end_time <= start_time:
            mock_end_time = start_time + 0.2 # Ensure end_time is definitely after start_time
        
        # We need at least 10 latency samples for percentile calculation to kick in.
        # Let's add some dummy latency values first.
        # It's better to test latency percentile calculations in a separate, focused test.
        for i in range(10):
             self.collector._record_latency(float(10 + i)) # Add some varied latency values

        with patch('time.time', return_value=mock_end_time):
            self.collector.end_request(request_id, success=True)
        
        self.assertNotIn(request_id, self.collector.current_requests)
        self.assertEqual(self.collector.metrics["inference.requests.active"], 0)
        self.assertEqual(self.collector.metrics["inference.requests.completed"], 1)
        self.assertEqual(self.collector.metrics["inference.requests.failed"], 0)

        # Check latency (this will be one of many samples, so direct check is hard)
        # Instead, check that latency history has one more entry
        self.assertTrue(len(self.collector.metrics_history.get("inference.latency.history", [])) > 10)
        
        # Check throughput (this also uses EMA, so direct check of one value is tricky)
        # We can check it's non-zero if tokens were generated
        if self.collector.current_requests.get(request_id, {}).get("tokens_generated", 0) > 0:
            self.assertTrue(self.collector.metrics["inference.throughput.tokens_per_second"] > 0)

    def test_failed_request(self):
        request_id = "test_req_fail"
        self.collector.start_request(request_id)
        self.collector.end_request(request_id, success=False)
        self.assertEqual(self.collector.metrics["inference.requests.failed"], 1)
        self.assertEqual(self.collector.metrics["inference.requests.completed"], 0) # Assuming this test runs after the successful one or is isolated

    def test_update_system_metrics(self):
        system_metrics = {
            "system.memory.used_mb": 1024,
            "system.cpu.utilization": 55.5
        }
        self.collector.update_system_metrics(system_metrics)
        self.assertEqual(self.collector.metrics["system.memory.used_mb"], 1024)
        self.assertEqual(self.collector.metrics["system.cpu.utilization"], 55.5)

    def test_update_batch_metrics(self):
        batch_metrics = {
            "batching.batch_size.avg": 8,
            "batching.queue_depth": 16
        }
        self.collector.update_batch_metrics(batch_metrics)
        self.assertEqual(self.collector.metrics["batching.batch_size.avg"], 8)
        self.assertEqual(self.collector.metrics["batching.queue_depth"], 16)

    @patch("builtins.open")
    @patch("os.path.join")
    @patch("json.dump")
    def test_export_all_metrics(self, mock_json_dump, mock_os_path_join, mock_open):
        # Configure mocks
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_os_path_join.return_value = "mocked_path/metrics_snapshot_mock_time.json"

        # Call the method
        self.collector._export_all_metrics()

        # Assertions
        mock_open.assert_called_once_with("mocked_path/metrics_snapshot_mock_time.json", "w")
        mock_json_dump.assert_called_once()
        # Check if the content being dumped is correct (simplified check)
        args, _ = mock_json_dump.call_args
        self.assertIn("timestamp", args[0])
        self.assertIn("metrics", args[0])
        self.assertEqual(args[0]["metrics"], self.collector.get_current_metrics())

    @patch("builtins.open")
    @patch("os.path.join")
    @patch("json.dumps") # Note: json.dumps for individual metric, not json.dump
    def test_export_metric(self, mock_json_dumps, mock_os_path_join, mock_open):
        # Configure mocks
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_os_path_join.return_value = "mocked_path/metrics_mock_date.jsonl"
        
        metric_data = InferenceMetric(name="test.export", value=123, timestamp=time.time())
        expected_json_str = '{"name": "test.export", "value": 123, "timestamp": ' + str(metric_data.timestamp) + ', "labels": {}}' # Simplified
        mock_json_dumps.return_value = expected_json_str # Mock the output of json.dumps

        # Call the method
        self.collector._export_metric(metric_data)

        # Assertions
        mock_os_path_join.assert_called_once() # Check that path join was called
        # Get the first argument of the call to os.path.join, which is the base path
        self.assertEqual(mock_os_path_join.call_args[0][0], self.export_path) 

        mock_open.assert_called_once_with("mocked_path/metrics_mock_date.jsonl", "a")
        mock_json_dumps.assert_called_once_with(metric_data.as_dict())
        mock_file.write.assert_called_once_with(expected_json_str + "\n")

if __name__ == '__main__':
    unittest.main() 