from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import os
import platform
import json
from enum import Enum
import subprocess

import numpy as np

logger = logging.getLogger(__name__)

# Try to import GPU-related libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    from metal_tools import metal_info
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

class AcceleratorType(Enum):
    """Types of accelerators supported by the system."""
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    INTEL_XPU = "intel_xpu"
    TPU = "tpu"
    UNKNOWN = "unknown"

class AcceleratorInfo:
    """Information about an accelerator device."""
    
    def __init__(self, 
                 device_id: int,
                 device_type: AcceleratorType,
                 name: str,
                 memory_gb: float,
                 compute_capability: Optional[str] = None,
                 additional_info: Optional[Dict[str, Any]] = None):
        """
        Initialize accelerator info.
        
        Args:
            device_id: Device identifier
            device_type: Type of accelerator
            name: Device name
            memory_gb: Total device memory in GB
            compute_capability: Compute capability (for CUDA devices)
            additional_info: Additional device-specific information
        """
        self.device_id = device_id
        self.device_type = device_type
        self.name = name
        self.memory_gb = memory_gb
        self.compute_capability = compute_capability
        self.additional_info = additional_info or {}
        
    def __str__(self):
        return f"{self.name} ({self.device_type.value}:{self.device_id}, {self.memory_gb:.2f} GB)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "name": self.name,
            "memory_gb": self.memory_gb,
            "compute_capability": self.compute_capability,
            "additional_info": self.additional_info
        }


class HardwareManager:
    """
    Hardware abstraction layer for managing different accelerator types.
    
    Provides a unified interface for device selection, optimization settings,
    and hardware-specific configuration.
    """
    
    def __init__(self, prefer_device_type: Optional[AcceleratorType] = None):
        """
        Initialize the hardware manager.
        
        Args:
            prefer_device_type: Preferred accelerator type to use
        """
        self.prefer_device_type = prefer_device_type
        self.devices = self._detect_devices()
        self.selected_device = self._select_best_device()
        
        # Optimization flags
        self.use_fp16 = self._should_use_fp16()
        self.use_bf16 = self._should_use_bf16() 
        self.use_amp = self._should_use_amp()
        self.use_trt = self._should_use_trt()
        
        # Log detected hardware
        self._log_hardware_info()
    
    def _detect_devices(self) -> Dict[AcceleratorType, List[AcceleratorInfo]]:
        """Detect available accelerator devices."""
        devices = {}
        
        # Always add CPU
        devices[AcceleratorType.CPU] = [
            AcceleratorInfo(
                device_id=0,
                device_type=AcceleratorType.CPU,
                name=platform.processor() or "CPU",
                memory_gb=self._get_system_memory_gb(),
                additional_info={
                    "cores": os.cpu_count(),
                    "architecture": platform.machine(),
                }
            )
        ]
        
        # Detect CUDA GPUs
        if TORCH_AVAILABLE and torch.cuda.is_available():
            cuda_devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                cuda_devices.append(
                    AcceleratorInfo(
                        device_id=i,
                        device_type=AcceleratorType.CUDA,
                        name=props.name,
                        memory_gb=props.total_memory / (1024**3),
                        compute_capability=f"{props.major}.{props.minor}",
                        additional_info={
                            "multi_processor_count": props.multi_processor_count,
                            "clock_rate": props.clock_rate,
                            "is_integrated": props.is_integrated,
                        }
                    )
                )
            devices[AcceleratorType.CUDA] = cuda_devices
        
        # Detect ROCm GPUs
        if TORCH_AVAILABLE and hasattr(torch, 'hip') and torch.hip.is_available():
            rocm_devices = []
            for i in range(torch.hip.device_count()):
                props = torch.hip.get_device_properties(i)
                rocm_devices.append(
                    AcceleratorInfo(
                        device_id=i,
                        device_type=AcceleratorType.ROCM,
                        name=props.name,
                        memory_gb=props.total_memory / (1024**3),
                        additional_info={
                            "clock_rate": props.clock_rate,
                        }
                    )
                )
            devices[AcceleratorType.ROCM] = rocm_devices
        
        # Detect Apple Metal (M-series)
        if METAL_AVAILABLE:
            try:
                metal_devices = []
                metal_info_data = metal_info()
                for i, device in enumerate(metal_info_data.get("devices", [])):
                    metal_devices.append(
                        AcceleratorInfo(
                            device_id=i,
                            device_type=AcceleratorType.METAL,
                            name=device.get("name", "Apple GPU"),
                            memory_gb=device.get("memory", 0) / 1024,
                            additional_info={
                                "renderer": device.get("renderer", ""),
                                "registry_id": device.get("registry_id", ""),
                            }
                        )
                    )
                if metal_devices:
                    devices[AcceleratorType.METAL] = metal_devices
            except Exception as e:
                logger.warning(f"Error detecting Metal devices: {str(e)}")
        
        # Detect TPUs
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.xla_multiprocessing as xmp
            
            def _get_tpu_info():
                import torch_xla
                devices = torch_xla.runtime.local_devices()
                return [d for d in devices if "TPU" in d]
            
            tpu_devices = []
            tpu_count = len(_get_tpu_info())
            
            if tpu_count > 0:
                # TPU details are limited, so we just count them
                for i in range(tpu_count):
                    tpu_devices.append(
                        AcceleratorInfo(
                            device_id=i,
                            device_type=AcceleratorType.TPU,
                            name=f"TPU v{3 if 'v3' in _get_tpu_info()[0] else 4}", # Rough estimation
                            memory_gb=16.0,  # Approximation
                            additional_info={"cores": tpu_count}
                        )
                    )
                devices[AcceleratorType.TPU] = tpu_devices
        except ImportError:
            pass
        
        return devices
    
    def _get_system_memory_gb(self) -> float:
        """Get total system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback for systems without psutil
            if platform.system() == "Linux":
                try:
                    with open('/proc/meminfo') as f:
                        for line in f:
                            if 'MemTotal' in line:
                                # Extract memory value (in kB)
                                return int(line.split()[1]) / (1024**2)
                except:
                    pass
            return 8.0  # Default assumption
    
    def _select_best_device(self) -> Tuple[AcceleratorType, int]:
        """
        Select the best device to use for inference.
        
        Returns:
            (device_type, device_id): Selected device
        """
        # Use preferred device type if specified and available
        if self.prefer_device_type and self.prefer_device_type in self.devices:
            return (self.prefer_device_type, 0)
        
        # Priority order for automatic selection
        priority = [
            AcceleratorType.CUDA,
            AcceleratorType.ROCM,
            AcceleratorType.METAL,
            AcceleratorType.TPU,
            AcceleratorType.INTEL_XPU,
            AcceleratorType.CPU
        ]
        
        for device_type in priority:
            if device_type in self.devices and self.devices[device_type]:
                # Select device with most memory
                best_device = max(self.devices[device_type], key=lambda d: d.memory_gb)
                return (device_type, best_device.device_id)
        
        # Default to CPU if nothing else available
        return (AcceleratorType.CPU, 0)
    
    def _should_use_fp16(self) -> bool:
        """Determine if FP16 precision should be used."""
        device_type, _ = self.selected_device
        
        if device_type == AcceleratorType.CUDA:
            # Check compute capability for FP16 support
            for device in self.devices[device_type]:
                if device.device_id == self.selected_device[1]:
                    if device.compute_capability:
                        major, minor = map(int, device.compute_capability.split('.'))
                        # Pascal (6.x) and above support FP16 well
                        return major >= 6
        
        elif device_type == AcceleratorType.METAL:
            # M-series chips support FP16 well
            return True
        
        return False
    
    def _should_use_bf16(self) -> bool:
        """Determine if BF16 precision should be used."""
        device_type, _ = self.selected_device
        
        if device_type == AcceleratorType.CUDA:
            # Check compute capability for BF16 support
            for device in self.devices[device_type]:
                if device.device_id == self.selected_device[1]:
                    if device.compute_capability:
                        major, minor = map(int, device.compute_capability.split('.'))
                        # Ampere (8.x) and above support BF16 well
                        return major >= 8
        
        elif device_type == AcceleratorType.TPU:
            # TPUs support BF16 natively
            return True
        
        return False
    
    def _should_use_amp(self) -> bool:
        """Determine if Automatic Mixed Precision should be used."""
        device_type, _ = self.selected_device
        return device_type in [AcceleratorType.CUDA, AcceleratorType.ROCM]
    
    def _should_use_trt(self) -> bool:
        """Determine if TensorRT should be used."""
        return (
            TRT_AVAILABLE and 
            self.selected_device[0] == AcceleratorType.CUDA and
            self._should_use_fp16()
        )
    
    def _log_hardware_info(self):
        """Log information about detected hardware."""
        logger.info(f"Detected hardware:")
        
        for device_type, devices in self.devices.items():
            for device in devices:
                logger.info(f"  - {device}")
        
        selected_type, selected_id = self.selected_device
        logger.info(f"Selected device: {selected_type.value}:{selected_id}")
        logger.info(f"Optimization flags: FP16={self.use_fp16}, BF16={self.use_bf16}, AMP={self.use_amp}, TRT={self.use_trt}")
    
    def get_device_str(self) -> str:
        """
        Get the device string for PyTorch.
        
        Returns:
            device_str: String in format "cuda:0", "cpu", etc.
        """
        device_type, device_id = self.selected_device
        
        if device_type == AcceleratorType.CUDA:
            return f"cuda:{device_id}"
        elif device_type == AcceleratorType.ROCM:
            return f"cuda:{device_id}"  # PyTorch uses "cuda" namespace for ROCm too
        elif device_type == AcceleratorType.METAL:
            return "mps"  # PyTorch Metal support
        else:
            return "cpu"
    
    def get_selected_device_info(self) -> Optional[AcceleratorInfo]:
        """Get information about the selected device."""
        device_type, device_id = self.selected_device
        
        if device_type in self.devices:
            for device in self.devices[device_type]:
                if device.device_id == device_id:
                    return device
        
        return None
    
    def get_all_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get information about all detected devices."""
        result = {}
        
        for device_type, devices in self.devices.items():
            result[device_type.value] = [device.to_dict() for device in devices]
        
        return result
    
    def optimize_model_for_device(self, model):
        """
        Optimize a PyTorch model for the selected device.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            optimized_model: Model optimized for the selected device
        """
        if not TORCH_AVAILABLE:
            return model
        
        device_str = self.get_device_str()
        logger.info(f"Optimizing model for {device_str}")
        
        # Move model to selected device
        model = model.to(device_str)
        
        # Apply precision optimizations
        if self.use_fp16:
            model = model.half()
        elif self.use_bf16 and hasattr(torch, 'bfloat16'):
            model = model.to(torch.bfloat16)
        
        # Set model to evaluation mode
        model.eval()
        
        # Apply JIT compilation if beneficial
        if self.selected_device[0] != AcceleratorType.CPU:
            try:
                # Use TorchScript for better performance
                example_input = self._create_example_input(model)
                if example_input is not None:
                    model = torch.jit.trace(model, example_input)
                    logger.info("Applied TorchScript optimization")
            except Exception as e:
                logger.warning(f"Failed to apply JIT optimization: {str(e)}")
        
        return model
    
    def _create_example_input(self, model):
        """Create example input for JIT tracing based on model architecture."""
        try:
            # This is highly model-dependent
            # Here we make a simple guess, but real implementation would need model metadata
            from inspect import signature
            
            # Get forward method signature
            sig = signature(model.forward)
            params = list(sig.parameters.values())
            
            if not params:
                return None
                
            # Assume first parameter is input tensor and guess shape
            batch_size = 1
            seq_len = 32
            
            # Check if model is a transformer
            if hasattr(model, 'config'):
                if hasattr(model.config, 'hidden_size'):
                    hidden_size = model.config.hidden_size
                    return torch.zeros(batch_size, seq_len, hidden_size, device=self.get_device_str())
            
            # Generic fallback for CNN-like models
            return torch.zeros(batch_size, 3, 224, 224, device=self.get_device_str())
            
        except Exception:
            return None