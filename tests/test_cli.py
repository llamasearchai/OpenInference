import pytest
from unittest.mock import patch
from OpenInference.cli import get_version, command_list_models
from OpenInference.models.registry import ModelRegistry
from OpenInference.hardware.accelerator import HardwareManager
import argparse

def test_get_version():
    # Assuming your __version__ is defined in OpenInference.__init__
    # and accessible for patching or direct import if needed.
    # For this example, let's assume it's '0.1.0' as in your setup.py
    # and __init__.py
    with patch('OpenInference.__version__', '0.1.0'):
        assert get_version() == '0.1.0'

@patch('OpenInference.cli.HardwareManager')
@patch('OpenInference.cli.ModelRegistry')
def test_command_list_models(MockModelRegistry, MockHardwareManager):
    # Setup mock instances
    mock_hw_manager = MockHardwareManager.return_value
    mock_model_registry = MockModelRegistry.return_value

    # Configure mock ModelRegistry to return some models
    mock_models_data = [
        {
            'name': 'test-model-1',
            'type': 'transformer',
            'size_mb': 100.0,
            'status': 'available',
            'is_loaded': False
        },
        {
            'name': 'test-model-2',
            'type': 'cnn',
            'size_mb': 50.0,
            'status': 'downloading',
            'is_loaded': False
        }
    ]
    mock_model_registry.list_available_models.return_value = mock_models_data

    # Setup args for the command
    args = argparse.Namespace(
        models_dir='dummay_models_dir',
        cache_dir='dummy_cache_dir',
        json=False
    )

    # Call the command function
    # We need to capture stdout to check the output
    with patch('builtins.print') as mock_print:
        command_list_models(args)

    # Assertions
    MockHardwareManager.assert_called_once()
    MockModelRegistry.assert_called_once_with(
        models_dir='dummay_models_dir',
        cache_dir='dummy_cache_dir',
        hardware_manager=mock_hw_manager
    )
    mock_model_registry.list_available_models.assert_called_once()

    # Check if print was called with expected model information
    # This is a basic check; more detailed checks can be added
    assert mock_print.call_count > 0
    call_args_list = [call[0][0] for call in mock_print.call_args_list if call[0]]
    
    assert "Available Models:" in call_args_list
    assert "Name: test-model-1" in call_args_list
    assert "Type: transformer" in call_args_list
    assert "Name: test-model-2" in call_args_list
    assert "Type: cnn" in call_args_list

@patch('OpenInference.cli.HardwareManager')
@patch('OpenInference.cli.ModelRegistry')
def test_command_list_models_json(MockModelRegistry, MockHardwareManager):
    mock_hw_manager = MockHardwareManager.return_value
    mock_model_registry = MockModelRegistry.return_value
    mock_models_data = [{'name': 'test-model-1', 'type': 'transformer', 'size_mb': 100.0, 'status': 'available', 'is_loaded': False}]
    mock_model_registry.list_available_models.return_value = mock_models_data

    args = argparse.Namespace(
        models_dir='models',
        cache_dir='.cache',
        json=True
    )

    with patch('builtins.print') as mock_print:
        with patch('OpenInference.cli.format_json') as mock_format_json:
            mock_format_json.return_value = "json_output"
            command_list_models(args)
    
    mock_format_json.assert_called_once_with(mock_models_data)
    mock_print.assert_called_once_with("json_output") 