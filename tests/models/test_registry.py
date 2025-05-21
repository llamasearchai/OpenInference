import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import json

from OpenInference.models.registry import ModelRegistry
from OpenInference.hardware.accelerator import HardwareManager

@patch('OpenInference.models.registry.HardwareManager')
@patch('os.makedirs') # Mock os.makedirs to avoid actual directory creation
@patch('os.path.exists') # Mock os.path.exists
def test_model_registry_initialization(mock_path_exists, mock_makedirs, MockHardwareManager):
    """Test the initialization of ModelRegistry."""
    mock_hw_manager_instance = MockHardwareManager.return_value
    models_dir = "custom_models_dir"
    cache_dir = "custom_cache_dir"

    # Simulate directories not existing initially, then existing after makedirs
    def path_exists_side_effect(path):
        if path in [models_dir, cache_dir]:
            # Simulate directory exists after os.makedirs is called on it
            return mock_makedirs.call_args and path in [ca[0][0] for ca in mock_makedirs.call_args_list]
        return False # Default for other paths if any
    mock_path_exists.side_effect = path_exists_side_effect

    registry = ModelRegistry(
        models_dir=models_dir,
        cache_dir=cache_dir,
        hardware_manager=mock_hw_manager_instance
    )

    assert registry.models_dir == models_dir
    assert registry.cache_dir == cache_dir
    assert registry.hardware_manager == mock_hw_manager_instance
    assert registry.model_configs == {}
    assert registry.loaded_models == {}
    assert registry.loaded_tokenizers == {}

    # Check that os.makedirs was called for models_dir and cache_dir
    mock_makedirs.assert_any_call(models_dir, exist_ok=True)
    mock_makedirs.assert_any_call(cache_dir, exist_ok=True)

    # Test with default dirs
    mock_makedirs.reset_mock()
    mock_path_exists.side_effect = lambda p: True # Assume default dirs exist
    default_registry = ModelRegistry(hardware_manager=mock_hw_manager_instance)
    assert default_registry.models_dir == "models"
    assert default_registry.cache_dir == ".model_cache"
    # No makedirs call if path_exists is True for them
    # However, the code calls makedirs with exist_ok=True anyway.
    mock_makedirs.assert_any_call("models", exist_ok=True)
    mock_makedirs.assert_any_call(".model_cache", exist_ok=True)

# Mock paths used by ModelRegistry as seen in main.py/cli.py
@patch('OpenInference.models.registry.os.path.exists')
@patch('OpenInference.models.registry.os.makedirs')
@patch('OpenInference.models.registry.HardwareManager') # Mock HardwareManager if it's a direct dependency
@patch('OpenInference.models.registry.ModelLoaderUtils.find_model_config_files') # Mock discovery
@patch('OpenInference.models.registry.ModelLoaderUtils.load_model_config') # Mock config loading
def test_model_registry_initialization_v2(
    mock_load_model_config,
    mock_find_configs,
    MockHardwareManager,
    mock_os_makedirs,
    mock_os_path_exists
):
    """Test initialization of ModelRegistry (as used in main.py/cli.py)."""
    mock_hw_manager_instance = MockHardwareManager.return_value
    models_dir = "test_models_dir"
    cache_dir = "test_cache_dir"

    mock_os_path_exists.return_value = True # Assume dirs exist or are created
    mock_find_configs.return_value = [] # No models discovered initially

    registry = ModelRegistry(
        models_dir=models_dir,
        cache_dir=cache_dir,
        hardware_manager=mock_hw_manager_instance
    )

    assert registry.models_dir == models_dir
    assert registry.cache_dir == cache_dir
    assert registry.hardware_manager == mock_hw_manager_instance
    assert registry.model_configs == {}
    assert registry.loaded_models == {}
    assert registry.loaded_tokenizers == {}

    mock_os_makedirs.assert_any_call(models_dir, exist_ok=True)
    mock_os_makedirs.assert_any_call(cache_dir, exist_ok=True)
    mock_find_configs.assert_called_once_with(models_dir)
    mock_load_model_config.assert_not_called() # Not called if no configs found

    # Test initialization with model discovery
    mock_find_configs.reset_mock()
    mock_load_model_config.reset_mock()
    
    config_file_path1 = os.path.join(models_dir, "model1", "config.json")
    config_file_path2 = os.path.join(models_dir, "model2", "config.json")
    mock_find_configs.return_value = [config_file_path1, config_file_path2]
    
    mock_config1 = {"name": "model1", "model_type": "bert"} # Simplified config
    mock_config2 = {"name": "model2", "model_type": "gpt"}
    
    # Side effect for loading multiple configs
    def load_config_side_effect(path):
        if path == config_file_path1: return mock_config1
        if path == config_file_path2: return mock_config2
        return None
    mock_load_model_config.side_effect = load_config_side_effect

    registry_with_models = ModelRegistry(
        models_dir=models_dir,
        cache_dir=cache_dir,
        hardware_manager=mock_hw_manager_instance
    )
    assert "model1" in registry_with_models.model_configs
    assert registry_with_models.model_configs["model1"] == mock_config1
    assert "model2" in registry_with_models.model_configs
    assert registry_with_models.model_configs["model2"] == mock_config2
    mock_load_model_config.assert_any_call(config_file_path1)
    mock_load_model_config.assert_any_call(config_file_path2)
    assert mock_load_model_config.call_count == 2 