import os
from typing import Dict, Any, List, Union, Optional, Tuple
import logging

import torch
import numpy as np

from .base import ModelLoader
from ...runtime.core import DeviceType

logger = logging.getLogger(__name__)

class PyTorchModelLoader(ModelLoader):
    """Loader for PyTorch models."""
    
    def __init__(self, device_type: DeviceType = DeviceType.CUDA, device_id: int = 0):
        super().__init__(device_type, device_id)
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """Get the appropriate torch device based on the configuration."""
        if self.device_type == DeviceType.CUDA:
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device(f"cuda:{self.device_id}")
        elif self.device_type == DeviceType.METAL:
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                logger.warning("Metal requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device("mps")
        return torch.device("cpu")
    
    def load(self, model_path: str, **kwargs) -> None:
        """Load a PyTorch model from the specified path."""
        try:
            if model_path.endswith(".pt") or model_path.endswith(".pth"):
                # Load state dict
                state_dict = torch.load(model_path, map_location="cpu")
                
                # Check if we have a model class specified
                model_class = kwargs.get("model_class")
                if model_class:
                    self.model = model_class(**kwargs.get("model_args", {}))
                    if "state_dict" in state_dict:
                        self.model.load_state_dict(state_dict["state_dict"])
                    else:
                        self.model.load_state_dict(state_dict)
                else:
                    # Try to load as TorchScript model
                    self.model = torch.jit.load(model_path, map_location="cpu")
            
            # Move model to the appropriate device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Store model configuration
            self.model_config = {
                "path": model_path,
                "device": str(self.device),
                **kwargs
            }
            
            logger.info(f"Loaded PyTorch model from {model_path} to {self.device}")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {str(e)}")
            raise
    
    def get_inputs_info(self) -> List[Dict[str, Any]]:
        """Get information about model inputs."""
        if not self.verify_model():
            return []
        
        # This is a simplified implementation - in practice, you would
        # want to introspect the model for accurate input information
        return [{"name": "input", "shape": "dynamic", "dtype": "float32"}]
    
    def get_outputs_info(self) -> List[Dict[str, Any]]:
        """Get information about model outputs."""
        if not self.verify_model():
            return []
        
        # This is a simplified implementation - in practice, you would
        # want to introspect the model for accurate output information
        return [{"name": "output", "shape": "dynamic", "dtype": "float32"}]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        if not self.verify_model():
            return {}
        
        metadata = {
            "framework": "pytorch",
            "device": str(self.device),
            "has_half_support": self.device.type != "cpu",
        }
        
        # Add info about model parameters if available
        if hasattr(self.model, "parameters"):
            num_params = sum(p.numel() for p in self.model.parameters())
            metadata["num_parameters"] = num_params
        
        return metadata
    
    def infer(self, inputs: Union[np.ndarray, torch.Tensor, Dict[str, Any]]) -> Any:
        """Run inference on the given inputs."""
        if not self.verify_model():
            raise ValueError("Model is not loaded")
        
        with torch.no_grad():
            # Convert inputs to tensor if necessary
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v.to(self.device)
                          for k, v in inputs.items()}
            elif isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            
            # Run inference
            outputs = self.model(inputs)
            
            # Convert outputs to numpy if needed
            if isinstance(outputs, torch.Tensor):
                return outputs.cpu().numpy()
            elif isinstance(outputs, dict):
                return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                        for k, v in outputs.items()}
            elif isinstance(outputs, (list, tuple)):
                return [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in outputs]
            
            return outputs