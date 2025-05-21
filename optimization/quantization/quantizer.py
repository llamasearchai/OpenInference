from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import logging
import os

import torch
import numpy as np

logger = logging.getLogger(__name__)

class ModelQuantizer(ABC):
    """Base class for model quantization."""
    
    def __init__(self, target_precision: str = "int8"):
        """
        Initialize the quantizer.
        
        Args:
            target_precision: The target precision after quantization.
                              Supported values: "int8", "int4", "fp16", "bf16"
        """
        self.target_precision = target_precision
        self.calibration_data = None
        self.quantized_model = None
    
    @abstractmethod
    def quantize(self, model: Any, **kwargs) -> Any:
        """Quantize the given model."""
        pass
    
    @abstractmethod
    def calibrate(self, calibration_data: Any) -> None:
        """Calibrate the quantization parameters using provided data."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the quantized model to the specified path."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> Any:
        """Load a quantized model from the specified path."""
        pass
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get statistics about the quantization process."""
        return {"target_precision": self.target_precision}