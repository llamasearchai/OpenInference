import pytest
from unittest.mock import patch, MagicMock

from OpenInference.main import OpenInference
from OpenInference.hardware.accelerator import AcceleratorType

@patch('OpenInference.main.HardwareManager')
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.MemoryManager')
@patch('OpenInference.main.KVCacheManager')
@patch('OpenInference.main.PerformanceTracker')
@patch('OpenInference.main.DynamicBatcher') # Assuming DynamicBatcher is in main or imported there
@patch('OpenInference.main.RuntimeContinuousBatcher') # Assuming RuntimeContinuousBatcher is in main
def test_openinference_initialization(
    MockRuntimeContinuousBatcher,
    MockDynamicBatcher,
    MockPerformanceTracker,
    MockKVCacheManager,
    MockMemoryManager,
    MockModelRegistry,
    MockHardwareManager
):
    """Test the initialization of the OpenInference class."""
    # Setup mock instances for managers
    mock_hw_manager = MockHardwareManager.return_value
    mock_hw_manager.get_device_str.return_value = 'cpu' # Example device

    mock_model_registry = MockModelRegistry.return_value
    mock_memory_manager = MockMemoryManager.return_value
    mock_kv_cache_manager = MockKVCacheManager.return_value
    mock_perf_tracker = MockPerformanceTracker.return_value
    mock_dynamic_batcher = MockDynamicBatcher.return_value

    # Initialize OpenInference
    engine = OpenInference(
        device='cpu',
        models_dir='test_models',
        cache_dir='test_cache',
        max_batch_size=16,
        enable_continuous_batching=False
    )

    # Assertions for initialization
    MockHardwareManager.assert_called_once_with(prefer_device_type=AcceleratorType.CPU)
    MockModelRegistry.assert_called_once_with(
        models_dir='test_models',
        cache_dir='test_cache',
        hardware_manager=mock_hw_manager
    )
    MockMemoryManager.assert_called_once_with(device='cpu')
    MockKVCacheManager.assert_called_once_with(device='cpu', max_memory_fraction=0.8)
    MockPerformanceTracker.assert_called_once_with(record_gpu_metrics=True, export_metrics=True)
    MockDynamicBatcher.assert_called_once_with(
        max_batch_size=16,
        max_wait_time_ms=100,
        performance_tracker=mock_perf_tracker
    )

    assert engine.hardware_manager == mock_hw_manager
    assert engine.model_registry == mock_model_registry
    assert engine.memory_manager == mock_memory_manager
    assert engine.kv_cache_manager == mock_kv_cache_manager
    assert engine.performance_tracker == mock_perf_tracker
    assert engine.dynamic_batcher == mock_dynamic_batcher
    assert engine.max_batch_size == 16
    assert not engine.enable_continuous_batching
    assert engine.loaded_models == {}
    assert engine.continuous_batchers == {}

    # Check logger call (optional, might need more specific patching)
    # For example, if logger.info is called, you can patch that specific logger 

@patch('OpenInference.main.HardwareManager')
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.MemoryManager')
@patch('OpenInference.main.KVCacheManager')
@patch('OpenInference.main.PerformanceTracker')
@patch('OpenInference.main.DynamicBatcher')
@patch('OpenInference.main.RuntimeContinuousBatcher')
@patch('OpenInference.main.PyTorchQuantizer') # Mock the quantizer
@patch('OpenInference.main.logger') # Mock the logger
def test_load_model_success(
    MockLogger,
    MockPyTorchQuantizer,
    MockRuntimeContinuousBatcher,
    MockDynamicBatcher,
    MockPerformanceTracker,
    MockKVCacheManager,
    MockMemoryManager,
    MockModelRegistry,
    MockHardwareManager
):
    """Test successful model loading, including quantization."""
    mock_hw_manager = MockHardwareManager.return_value
    mock_model_registry = MockModelRegistry.return_value
    mock_quantizer_instance = MockPyTorchQuantizer.return_value

    # Setup engine
    engine = OpenInference(device='cpu', enable_continuous_batching=False)
    engine.hardware_manager = mock_hw_manager # Ensure using the patched one
    engine.model_registry = mock_model_registry

    # Mock model, config, and tokenizer
    mock_model = MagicMock()
    mock_config = MagicMock()
    mock_config.max_position_embeddings = 2048 # For continuous batcher path
    mock_config.model_type = 'gpt' # To trigger _is_transformer_model
    mock_tokenizer = MagicMock()
    
    mock_model_registry.load_model.return_value = mock_model
    mock_model_registry.get_model_config.return_value = mock_config
    mock_model_registry.get_tokenizer.return_value = mock_tokenizer
    mock_hw_manager.optimize_model_for_device.return_value = mock_model
    mock_quantizer_instance.quantize.return_value = mock_model # Quantizer returns the (mock) model

    model_name = "test_model"

    # Test loading without quantization
    success = engine.load_model(model_name)
    assert success
    assert model_name in engine.loaded_models
    assert engine.loaded_models[model_name]["model"] == mock_model
    mock_model_registry.load_model.assert_called_with(model_name)
    mock_hw_manager.optimize_model_for_device.assert_called_with(mock_model)
    MockPyTorchQuantizer.assert_not_called() # Not called if quantize is None
    MockLogger.info.assert_any_call(f"Loading model: {model_name}")
    MockLogger.info.assert_any_call(f"Model {model_name} loaded successfully")

    # Reset mocks for quantization test
    engine.loaded_models = {} # Clear loaded models
    MockPyTorchQuantizer.reset_mock()
    mock_model_registry.load_model.reset_mock()
    mock_hw_manager.optimize_model_for_device.reset_mock()
    MockLogger.reset_mock()

    # Test loading with quantization
    quantize_level = "int8"
    success_quantized = engine.load_model(model_name, quantize=quantize_level)
    assert success_quantized
    assert model_name in engine.loaded_models
    MockPyTorchQuantizer.assert_called_once_with(target_precision=quantize_level)
    mock_quantizer_instance.quantize.assert_called_once_with(mock_model)
    MockLogger.info.assert_any_call(f"Applying {quantize_level} quantization to model")

@patch('OpenInference.main.HardwareManager')
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.MemoryManager')
@patch('OpenInference.main.KVCacheManager')
@patch('OpenInference.main.PerformanceTracker')
@patch('OpenInference.main.DynamicBatcher')
@patch('OpenInference.main.logger') # Mock the logger
def test_load_model_failure( 
    MockLogger,
    MockDynamicBatcher, 
    MockPerformanceTracker, 
    MockKVCacheManager, 
    MockMemoryManager, 
    MockModelRegistry, 
    MockHardwareManager
):
    """Test model loading failure when model_registry.load_model returns None."""
    mock_model_registry = MockModelRegistry.return_value
    engine = OpenInference(device='cpu')
    engine.model_registry = mock_model_registry # Ensure using the patched one

    mock_model_registry.load_model.return_value = None
    model_name = "non_existent_model"

    success = engine.load_model(model_name)

    assert not success
    assert model_name not in engine.loaded_models
    MockLogger.error.assert_any_call(f"Failed to load model: {model_name}")

@patch('OpenInference.main.HardwareManager')
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.MemoryManager')
@patch('OpenInference.main.KVCacheManager')
@patch('OpenInference.main.PerformanceTracker')
@patch('OpenInference.main.DynamicBatcher')
@patch('OpenInference.main.RuntimeContinuousBatcher')
@patch('OpenInference.main.logger')
def test_load_model_continuous_batching_setup(
    MockLogger,
    MockRuntimeContinuousBatcher,
    MockDynamicBatcher,
    MockPerformanceTracker,
    MockKVCacheManager,
    MockMemoryManager,
    MockModelRegistry,
    MockHardwareManager
):
    """Test that RuntimeContinuousBatcher is set up for transformer models if enabled."""
    mock_hw_manager = MockHardwareManager.return_value
    mock_hw_manager.get_device_str.return_value = 'cuda:0'
    mock_model_registry = MockModelRegistry.return_value
    mock_continuous_batcher_instance = MockRuntimeContinuousBatcher.return_value

    engine = OpenInference(device='cuda', enable_continuous_batching=True, max_batch_size=8)
    engine.hardware_manager = mock_hw_manager
    engine.model_registry = mock_model_registry

    mock_model = MagicMock()
    mock_config = MagicMock()
    mock_config.max_position_embeddings = 1024
    mock_config.model_type = 'llama' # Transformer type
    mock_tokenizer = MagicMock()

    mock_model_registry.load_model.return_value = mock_model
    mock_model_registry.get_model_config.return_value = mock_config
    mock_model_registry.get_tokenizer.return_value = mock_tokenizer
    mock_hw_manager.optimize_model_for_device.return_value = mock_model

    model_name = "transformer_model"
    success = engine.load_model(model_name)

    assert success
    assert model_name in engine.continuous_batchers
    MockRuntimeContinuousBatcher.assert_called_once_with(
        model=mock_model,
        device='cuda:0',
        max_batch_size=8,
        max_input_length=1024,
        max_prefill_tokens=4096,
        max_attention_window=1024,
    )
    mock_continuous_batcher_instance.start.assert_called_once() 

@patch('OpenInference.main.HardwareManager')
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.MemoryManager')
@patch('OpenInference.main.KVCacheManager')
@patch('OpenInference.main.PerformanceTracker')
@patch('OpenInference.main.DynamicBatcher')
@patch('OpenInference.main.RuntimeContinuousBatcher')
@patch('OpenInference.main.logger')
@patch('gc.collect') # Mock garbage collection
def test_unload_model(
    MockGcCollect,
    MockLogger,
    MockRuntimeContinuousBatcher, # Keep for consistency even if not directly used in all paths
    MockDynamicBatcher,
    MockPerformanceTracker,
    MockKVCacheManager,
    MockMemoryManager,
    MockModelRegistry,
    MockHardwareManager
):
    """Test model unloading functionality."""
    mock_hw_manager = MockHardwareManager.return_value
    mock_memory_manager_instance = MockMemoryManager.return_value
    engine = OpenInference(device='cpu', enable_continuous_batching=True)
    engine.hardware_manager = mock_hw_manager
    engine.memory_manager = mock_memory_manager_instance # Ensure using the patched one

    model_name = "test_model_to_unload"
    mock_model_data = {"model": MagicMock(), "config": MagicMock(), "tokenizer": MagicMock()}

    # Case 1: Unload a loaded model (without continuous batcher)
    engine.loaded_models[model_name] = mock_model_data
    success_unload = engine.unload_model(model_name)
    assert success_unload
    assert model_name not in engine.loaded_models
    MockLogger.info.assert_any_call(f"Model {model_name} unloaded successfully")
    MockGcCollect.assert_called_once() # gc.collect should be called
    if hasattr(mock_memory_manager_instance, 'clear_cache'):
        mock_memory_manager_instance.clear_cache.assert_called_once()
    MockGcCollect.reset_mock() # Reset for next test case
    if hasattr(mock_memory_manager_instance, 'clear_cache'):
      mock_memory_manager_instance.clear_cache.reset_mock()
    MockLogger.reset_mock()

    # Case 2: Unload a loaded model (with continuous batcher)
    mock_batcher_instance = MagicMock()
    engine.loaded_models[model_name] = mock_model_data
    engine.continuous_batchers[model_name] = mock_batcher_instance
    success_unload_with_batcher = engine.unload_model(model_name)
    assert success_unload_with_batcher
    assert model_name not in engine.loaded_models
    assert model_name not in engine.continuous_batchers
    mock_batcher_instance.stop.assert_called_once()
    MockLogger.info.assert_any_call(f"Model {model_name} unloaded successfully")
    MockGcCollect.assert_called_once() 
    if hasattr(mock_memory_manager_instance, 'clear_cache'):
        mock_memory_manager_instance.clear_cache.assert_called_once()
    MockGcCollect.reset_mock()
    if hasattr(mock_memory_manager_instance, 'clear_cache'):
      mock_memory_manager_instance.clear_cache.reset_mock()
    MockLogger.reset_mock()

    # Case 3: Attempt to unload a model that is not loaded
    non_loaded_model_name = "not_loaded_model"
    success_not_loaded = engine.unload_model(non_loaded_model_name)
    assert not success_not_loaded
    MockLogger.warning.assert_any_call(f"Model {non_loaded_model_name} not loaded")
    MockGcCollect.assert_not_called() # Should not be called if model not found
    if hasattr(mock_memory_manager_instance, 'clear_cache'):
        mock_memory_manager_instance.clear_cache.assert_not_called()
    MockLogger.reset_mock()

    # Case 4: Error during unloading (e.g., batcher.stop() raises an exception)
    engine.loaded_models[model_name] = mock_model_data
    mock_error_batcher = MagicMock()
    mock_error_batcher.stop.side_effect = Exception("Batcher stop error")
    engine.continuous_batchers[model_name] = mock_error_batcher
    
    success_error = engine.unload_model(model_name)
    assert not success_error
    # Model might still be in loaded_models if error happens before removal
    # depending on exact implementation, or it might be removed.
    # For this test, we check the logger for the error.
    MockLogger.error.assert_any_call(f"Error unloading model {model_name}: Batcher stop error") 

@patch('OpenInference.main.HardwareManager') # Minimal mocks needed for this helper
def test_is_transformer_model(MockHardwareManager):
    engine = OpenInference(device='cpu')
    model_name = "test_model_type"

    # Case 1: Model not loaded
    assert not engine._is_transformer_model(model_name)

    # Case 2: Model loaded, config has a known transformer model_type
    mock_config_gpt = MagicMock()
    mock_config_gpt.model_type = 'gpt2'
    engine.loaded_models[model_name] = {"config": mock_config_gpt}
    assert engine._is_transformer_model(model_name)

    mock_config_llama = MagicMock()
    mock_config_llama.model_type = 'llama'
    engine.loaded_models[model_name] = {"config": mock_config_llama}
    assert engine._is_transformer_model(model_name)

    # Case 3: Model loaded, config has a non-transformer model_type
    mock_config_cnn = MagicMock()
    mock_config_cnn.model_type = 'resnet'
    engine.loaded_models[model_name] = {"config": mock_config_cnn}
    assert not engine._is_transformer_model(model_name)

    # Case 4: Model loaded, config has model_type with partial match
    mock_config_partial = MagicMock()
    mock_config_partial.model_type = 'custom_gpt_model'
    engine.loaded_models[model_name] = {"config": mock_config_partial}
    assert engine._is_transformer_model(model_name) # 'gpt' is in 'custom_gpt_model'

    # Case 5: Model loaded, no model_type, but has other transformer attributes
    mock_config_attrs = MagicMock()
    # del mock_config_attrs.model_type # Ensure model_type is not present
    # For MagicMock, if an attribute is not set, trying to access it raises an AttributeError.
    # To simulate it not being there, we can ensure it's not set or set it to a special value
    # and check for that value in the code if needed, or rely on hasattr.
    # The current _is_transformer_model implementation uses hasattr for these.
    mock_config_attrs.hidden_size = 768
    mock_config_attrs.num_hidden_layers = 12
    mock_config_attrs.num_attention_heads = 12
    # Clear model_type by making it not exist on the mock
    # One way: configure it to raise AttributeError when accessed
    type(mock_config_attrs).model_type = MagicMock(side_effect=AttributeError)
    engine.loaded_models[model_name] = {"config": mock_config_attrs}
    assert engine._is_transformer_model(model_name)

    # Case 6: Model loaded, no model_type, missing some transformer attributes
    mock_config_missing_attrs = MagicMock()
    type(mock_config_missing_attrs).model_type = MagicMock(side_effect=AttributeError)
    mock_config_missing_attrs.hidden_size = 768
    # num_hidden_layers is missing
    mock_config_missing_attrs.num_attention_heads = 12
    engine.loaded_models[model_name] = {"config": mock_config_missing_attrs}
    assert not engine._is_transformer_model(model_name)

    # Clean up
    del engine.loaded_models[model_name]

@patch('OpenInference.main.HardwareManager')
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.MemoryManager')
@patch('OpenInference.main.KVCacheManager')
@patch('OpenInference.main.PerformanceTracker')
@patch('OpenInference.main.DynamicBatcher')
@patch('OpenInference.main.RuntimeContinuousBatcher')
@patch('OpenInference.main.logger')
@patch('torch.no_grad') # Mock torch.no_grad context manager
@patch('time.time') # Mock time.time for compute_start_time
def test_run_inference_non_streaming_direct_compute(
    MockTime,
    MockTorchNoGrad,
    MockLogger,
    MockRuntimeCB, MockDynamicBatcher, MockPerfTracker, MockKVCache, MockMemManager, MockModelRegistry, MockHWManager
):
    """Test run_inference: non-streaming, no batch_size > 1 (direct computation)."""
    # Setup mocks
    mock_perf_tracker_instance = MockPerfTracker.return_value
    MockTime.return_value = 123.456 # Consistent time for testing

    engine = OpenInference(device='cpu', enable_continuous_batching=False)
    engine.performance_tracker = mock_perf_tracker_instance

    model_name = "test_model_direct"
    mock_model = MagicMock()
    mock_inputs = MagicMock()
    mock_outputs = MagicMock()
    mock_model.return_value = mock_outputs # Model invocation returns mock_outputs
    mock_inputs.shape = [1, 10] # Example shape
    mock_outputs.shape = [1, 5] # Example output shape

    engine.loaded_models[model_name] = {"model": mock_model, "config": MagicMock(), "tokenizer": MagicMock()}

    # Call run_inference
    kwargs_passthrough = {"param1": "value1"}
    result = engine.run_inference(model_name, mock_inputs, **kwargs_passthrough)

    # Assertions
    assert result == mock_outputs
    mock_perf_tracker_instance.start_request.assert_called_once()
    tracking_info = mock_perf_tracker_instance.start_request.return_value 
    
    MockTorchNoGrad.assert_called_once() # torch.no_grad should wrap the model call
    mock_model.assert_called_once_with(mock_inputs, **kwargs_passthrough)
    # Check if inputs were moved to device (if model has device and inputs have .to)
    if hasattr(mock_inputs, 'to') and hasattr(mock_model, 'device'):
        mock_inputs.to.assert_called_with(mock_model.device)

    mock_perf_tracker_instance.finish_request.assert_called_once_with(
        tracking_info=tracking_info,
        batch_size=1, # Direct compute, so batch size is 1
        input_shape=[1,10],
        output_shape=[1,5],
        compute_start_time=123.456, # From MockTime
        success=True
    )
    MockDynamicBatcher.return_value.process.assert_not_called()

@patch('OpenInference.main.HardwareManager')
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.MemoryManager')
@patch('OpenInference.main.KVCacheManager')
@patch('OpenInference.main.PerformanceTracker')
@patch('OpenInference.main.DynamicBatcher')
@patch('OpenInference.main.RuntimeContinuousBatcher')
@patch('OpenInference.main.logger')
def test_run_inference_non_streaming_dynamic_batching(
    MockLogger,
    MockRuntimeCB, MockDynamicBatcher, MockPerfTracker, MockKVCache, MockMemManager, MockModelRegistry, MockHWManager
):
    """Test run_inference: non-streaming, with batch_size > 1 (DynamicBatcher)."""
    mock_perf_tracker_instance = MockPerfTracker.return_value
    mock_dynamic_batcher_instance = MockDynamicBatcher.return_value
    
    engine = OpenInference(device='cpu', enable_continuous_batching=False, max_batch_size=32)
    engine.performance_tracker = mock_perf_tracker_instance
    engine.dynamic_batcher = mock_dynamic_batcher_instance

    model_name = "test_model_batch"
    mock_model = MagicMock()
    mock_inputs = MagicMock()
    mock_batch_outputs = MagicMock()

    engine.loaded_models[model_name] = {"model": mock_model, "config": MagicMock(), "tokenizer": MagicMock()}
    mock_dynamic_batcher_instance.process.return_value = mock_batch_outputs

    # Call run_inference with a batch_size override
    batch_size_override = 4
    kwargs_passthrough = {"param2": "value2"}
    result = engine.run_inference(model_name, mock_inputs, batch_size=batch_size_override, **kwargs_passthrough)

    assert result == mock_batch_outputs
    mock_perf_tracker_instance.start_request.assert_called_once()
    # DynamicBatcher is responsible for calling finish_request internally via compute_fn
    # So we don't check finish_request directly on engine's tracker here for the main call.
    
    # The compute_fn passed to dynamic_batcher.process will be called by the batcher.
    # We need to check that dynamic_batcher.process was called correctly.
    # The first arg to process is `inputs`, the second is `compute_fn`, third is `batch_size`.
    # We can capture the compute_fn and test it separately if needed, but for now, just check the call.
    assert mock_dynamic_batcher_instance.process.call_count == 1
    call_args = mock_dynamic_batcher_instance.process.call_args
    assert call_args[1]['inputs'] == mock_inputs
    assert call_args[1]['batch_size'] == batch_size_override
    # compute_fn is harder to assert directly without more complex arg capture

@patch('OpenInference.main.logger')
def test_run_inference_model_not_loaded(MockLogger):
    engine = OpenInference(device='cpu')
    result = engine.run_inference("unloaded_model", MagicMock())
    assert result is None
    MockLogger.error.assert_called_once_with("Model unloaded_model not loaded") 

@patch('OpenInference.main.HardwareManager')
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.MemoryManager')
@patch('OpenInference.main.KVCacheManager')
@patch('OpenInference.main.PerformanceTracker')
@patch('OpenInference.main.DynamicBatcher')
@patch('OpenInference.main.RuntimeContinuousBatcher')
@patch('OpenInference.main.logger')
@patch('OpenInference.main.queue.Queue') # Mock the queue.Queue
def test_run_inference_streaming_success(
    MockQueue,
    MockLogger,
    MockRuntimeCB, MockDynamicBatcher, MockPerfTracker, MockKVCache, MockMemManager, MockModelRegistry, MockHWManager
):
    """Test run_inference: streaming path with RuntimeContinuousBatcher success."""
    mock_perf_tracker_instance = MockPerfTracker.return_value
    mock_continuous_batcher_instance = MockRuntimeCB.return_value
    mock_queue_instance = MockQueue.return_value

    engine = OpenInference(device='cuda', enable_continuous_batching=True)
    engine.performance_tracker = mock_perf_tracker_instance

    model_name = "test_stream_model"
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3] # Encoded prompt tokens
    mock_tokenizer.decode.side_effect = lambda ids, skip_special_tokens: f"text_for_ids_{ids}" 

    engine.loaded_models[model_name] = {
        "model": MagicMock(), 
        "config": MagicMock(max_length=256), 
        "tokenizer": mock_tokenizer
    }
    engine.continuous_batchers[model_name] = mock_continuous_batcher_instance

    inputs = "This is a test prompt."
    kwargs_passthrough = {"max_new_tokens": 50, "temperature": 0.7}

    # Simulate items being put on the queue by the batcher's callback
    # (token_ids, is_done)
    queue_items = [
        ([10], False), # Chunk 1
        ([20, 30], False), # Chunk 2
        ([], True) # Done
    ]
    mock_queue_instance.get.side_effect = queue_items

    # Call run_inference for streaming
    generator = engine.run_inference(model_name, inputs, stream=True, **kwargs_passthrough)

    # Consume the generator and check results
    results = [item for item in generator]
    expected_results = ["text_for_ids_[10]", "text_for_ids_[20, 30]"]
    assert results == expected_results

    # Assertions
    mock_perf_tracker_instance.start_request.assert_called_once()
    tracking_info = mock_perf_tracker_instance.start_request.return_value
    mock_tokenizer.encode.assert_called_once_with(inputs, add_special_tokens=True)
    
    # Check submit_request call on the continuous batcher
    mock_continuous_batcher_instance.submit_request.assert_called_once()
    submit_call_args = mock_continuous_batcher_instance.submit_request.call_args
    assert submit_call_args[1]['prompt_tokens'] == [1, 2, 3]
    assert submit_call_args[1]['max_new_tokens'] == 50
    assert submit_call_args[1]['temperature'] == 0.7
    assert callable(submit_call_args[1]['callback'])

    # Simulate callback invocation to ensure queue.put is called
    # This part is a bit tricky as the callback is defined inside run_inference.
    # We can verify it by checking `mock_queue_instance.put` calls if needed,
    # but the `mock_queue_instance.get.side_effect` above already simulates the flow.

    # Check that performance tracker is finished after generator is exhausted
    # This happens in the `finally` block of the generator
    mock_perf_tracker_instance.finish_request.assert_called_once_with(
        tracking_info=tracking_info,
        success=True 
    )
    MockQueue.assert_called_once() # Ensure a queue was created

@patch('OpenInference.main.logger')
def test_run_inference_streaming_no_tokenizer(MockLogger):
    engine = OpenInference(device='cuda', enable_continuous_batching=True)
    model_name = "stream_no_tokenizer"
    engine.loaded_models[model_name] = {"model": MagicMock(), "config": MagicMock(), "tokenizer": None} # No tokenizer
    engine.continuous_batchers[model_name] = MagicMock() # Has a batcher
    engine.performance_tracker = MagicMock()

    result = engine.run_inference(model_name, "prompt", stream=True)
    assert result is None
    MockLogger.error.assert_any_call(f"Tokenizer not found for model {model_name}, cannot stream.")
    engine.performance_tracker.finish_request.assert_called_once_with(tracking_info=engine.performance_tracker.start_request.return_value, success=False)

@patch('OpenInference.main.logger')
def test_run_inference_streaming_tokenizer_encode_error(MockLogger):
    engine = OpenInference(device='cuda', enable_continuous_batching=True)
    model_name = "stream_encode_error"
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = Exception("Encoding failed")
    engine.loaded_models[model_name] = {"model": MagicMock(), "config": MagicMock(), "tokenizer": mock_tokenizer}
    engine.continuous_batchers[model_name] = MagicMock()
    engine.performance_tracker = MagicMock()

    result = engine.run_inference(model_name, "prompt", stream=True)
    assert result is None
    MockLogger.error.assert_any_call(f"Error tokenizing input for model {model_name}: Encoding failed")
    engine.performance_tracker.finish_request.assert_called_once_with(tracking_info=engine.performance_tracker.start_request.return_value, success=False) 

@patch('OpenInference.main.HardwareManager')
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.MemoryManager')
@patch('OpenInference.main.KVCacheManager')
@patch('OpenInference.main.PerformanceTracker')
@patch('OpenInference.main.DynamicBatcher')
@patch('OpenInference.main.RuntimeContinuousBatcher')
@patch('OpenInference.main.logger')
def test_get_performance_metrics(
    MockLogger, MockRuntimeCB, MockDynamicBatcher, MockPerfTracker, 
    MockKVCache, MockMemManager, MockModelRegistry, MockHWManager
):
    """Test the get_performance_metrics method."""
    mock_perf_tracker_instance = MockPerfTracker.return_value
    mock_hw_manager_instance = MockHWManager.return_value
    mock_mem_manager_instance = MockMemManager.return_value
    mock_kv_cache_instance = MockKVCache.return_value
    mock_dynamic_batcher_instance = MockDynamicBatcher.return_value

    engine = OpenInference(device='cpu')
    engine.performance_tracker = mock_perf_tracker_instance
    engine.hardware_manager = mock_hw_manager_instance
    engine.memory_manager = mock_mem_manager_instance
    engine.kv_cache_manager = mock_kv_cache_instance
    engine.dynamic_batcher = mock_dynamic_batcher_instance

    # Mock return values for individual stat getters
    mock_perf_tracker_instance.get_metrics_summary.return_value = {"perf_summary": "data"}
    mock_selected_device_info = MagicMock()
    mock_selected_device_info.to_dict.return_value = {"hw_info": "details"}
    mock_hw_manager_instance.get_selected_device_info.return_value = mock_selected_device_info
    mock_mem_manager_instance.get_memory_stats.return_value = {"mem_stats": "usage"}
    mock_kv_cache_instance.get_stats.return_value = {"kv_stats": "cache_info"}
    mock_dynamic_batcher_instance.get_stats.return_value = {"dyn_batch_stats": "batcher_data"}

    # Case 1: No continuous batchers
    metrics = engine.get_performance_metrics()
    assert metrics["perf_summary"] == "data"
    assert metrics["hardware"] == {"hw_info": "details"}
    assert metrics["memory"] == {"mem_stats": "usage"}
    assert metrics["kv_cache"] == {"kv_stats": "cache_info"}
    assert metrics["dynamic_batcher"] == {"dyn_batch_stats": "batcher_data"}
    assert "continuous_batchers" not in metrics

    # Case 2: With continuous batchers
    mock_cb1 = MagicMock()
    mock_cb1.get_stats.return_value = {"cb1_stats": "cb1_data"}
    mock_cb2 = MagicMock()
    mock_cb2.get_stats.return_value = {"cb2_stats": "cb2_data"}
    engine.continuous_batchers = {"model1": mock_cb1, "model2": mock_cb2}
    
    metrics_with_cb = engine.get_performance_metrics()
    assert metrics_with_cb["continuous_batchers"] == {
        "model1": {"cb1_stats": "cb1_data"},
        "model2": {"cb2_stats": "cb2_data"}
    }
    mock_cb1.get_stats.assert_called_once()
    mock_cb2.get_stats.assert_called_once()


@patch('OpenInference.main.HardwareManager')
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.MemoryManager')
@patch('OpenInference.main.KVCacheManager')
@patch('OpenInference.main.PerformanceTracker')
@patch('OpenInference.main.DynamicBatcher')
@patch('OpenInference.main.RuntimeContinuousBatcher')
@patch('OpenInference.main.logger')
def test_shutdown(
    MockLogger, MockRuntimeCB, MockDynamicBatcher, MockPerfTracker, 
    MockKVCache, MockMemManager, MockModelRegistry, MockHWManager
):
    """Test the shutdown method."""
    mock_perf_tracker_instance = MockPerfTracker.return_value
    mock_mem_manager_instance = MockMemManager.return_value
    mock_kv_cache_instance = MockKVCache.return_value

    engine = OpenInference(device='cpu')
    engine.performance_tracker = mock_perf_tracker_instance
    engine.memory_manager = mock_mem_manager_instance
    engine.kv_cache_manager = mock_kv_cache_instance

    # Mock loaded models and continuous batchers
    mock_cb1 = MagicMock()
    mock_cb2 = MagicMock()
    engine.continuous_batchers = {"model1_cb": mock_cb1, "model2_cb": mock_cb2}
    
    # Use a copy of keys for loaded_models as unload_model modifies the dict
    engine.loaded_models = {"model1_cb": {}, "modelA": {}, "model2_cb": {}, "modelB": {}}
    # Mock unload_model to prevent actual logic and just check calls
    engine.unload_model = MagicMock(return_value=True) 

    engine.shutdown()

    # Assertions for continuous batchers
    mock_cb1.stop.assert_called_once()
    mock_cb2.stop.assert_called_once()

    # Assertions for unload_model calls
    # Order of unload_model calls might not be guaranteed, so check counts and individual calls
    assert engine.unload_model.call_count == 4
    engine.unload_model.assert_any_call("model1_cb")
    engine.unload_model.assert_any_call("modelA")
    engine.unload_model.assert_any_call("model2_cb")
    engine.unload_model.assert_any_call("modelB")

    # Assertions for other manager cleanups
    mock_perf_tracker_instance.stop.assert_called_once()
    mock_mem_manager_instance.clear_all.assert_called_once()
    mock_kv_cache_instance.clear_all.assert_called_once()
    MockLogger.info.assert_any_call("OpenInference system shutdown complete")


@patch('OpenInference.main.start_server')
@patch('OpenInference.main.HardwareManager') # To avoid full engine setup for this simple test
@patch('OpenInference.main.ModelRegistry')
@patch('OpenInference.main.PerformanceTracker')
def test_start_server_passthrough(
    MockPerfTracker, MockModelRegistry, MockHWManager, MockStartServer
):
    """Test that start_server calls the API server function with correct args."""
    engine = OpenInference() # Initialize with defaults
    # Replace managers with mocks if they were not fully mocked by class decorators
    engine.hardware_manager = MockHWManager.return_value
    engine.model_registry = MockModelRegistry.return_value
    engine.performance_tracker = MockPerfTracker.return_value

    host = "127.0.0.1"
    port = 9090
    workers = 2

    engine.start_server(host=host, port=port, workers=workers)

    MockStartServer.assert_called_once_with(
        host=host,
        port=port,
        workers=workers,
        model_registry=engine.model_registry,
        hardware_manager=engine.hardware_manager,
        performance_tracker=engine.performance_tracker,
        inference_engine=engine
    ) 