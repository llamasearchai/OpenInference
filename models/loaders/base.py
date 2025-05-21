from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import os
import logging

import torch
import numpy as np

from ...runtime.core import DeviceType

logger = logging.getLogger(__name__)

class ModelLoader(ABC):
    """Base class for all model loaders."""
    
    def __init__(self, device_type: DeviceType = DeviceType.CUDA, device_id: int = 0):
        self.device_type = device_type
        self.device_id = device_id
        self.model = None
        self.model_config = {}
    
    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """Load a model from the specified path."""
        pass
    
    @abstractmethod
    def get_inputs_info(self) -> List[Dict[str, Any]]:
        """Get information about model inputs."""
        pass
    
    @abstractmethod
    def get_outputs_info(self) -> List[Dict[str, Any]]:
        """Get information about model outputs."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        pass
    
    @abstractmethod
    def infer(self, inputs: Union[np.ndarray, torch.Tensor, Dict[str, Any]]) -> Any:
        """Run inference on the given inputs."""
        pass
    
    def verify_model(self) -> bool:
        """Verify that the model is valid and can be used for inference."""
        if self.model is None:
            logger.error("Model has not been loaded yet")
            return False
        return True