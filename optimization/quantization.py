"""
Quantization module for optimizing model size and inference speed.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, Union, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, quantization features will be disabled")

class PyTorchQuantizer:
    """
    Quantizer for PyTorch models.
    
    Supports INT8, INT4, FP16, and BF16 quantization for improved
    performance and reduced memory footprint.
    """
    
    def __init__(self, 
                target_precision: str = "int8",
                calibration_samples: int = 100,
                per_channel: bool = True,
                static_quantization: bool = True):
        """
        Initialize the PyTorch quantizer.
        
        Args:
            target_precision: Target precision ("int8", "int4", "fp16", "bf16")
            calibration_samples: Number of samples to use for calibration
            per_channel: Whether to use per-channel quantization
            static_quantization: Whether to use static or dynamic quantization
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for quantization")
        
        self.target_precision = target_precision.lower()
        self.calibration_samples = calibration_samples
        self.per_channel = per_channel
        self.static_quantization = static_quantization
        
        # Stats for before/after comparison
        self.stats = {
            "original_size_bytes": 0,
            "quantized_size_bytes": 0,
            "size_reduction_percent": 0.0,
            "quantization_time_seconds": 0.0
        }
        
        # Verify target precision is supported
        supported_precisions = ["int8", "int4", "fp16", "bf16"]
        if self.target_precision not in supported_precisions:
            raise ValueError(f"Unsupported precision: {target_precision}. "
                            f"Supported precisions: {supported_precisions}")
    
    def quantize(self, model: 'torch.nn.Module') -> 'torch.nn.Module':
        """
        Quantize a PyTorch model.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            quantized_model: Quantized PyTorch model
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot quantize model")
            return model
        
        start_time = time.time()
        logger.info(f"Starting quantization to {self.target_precision}")
        
        # Estimate original model size
        original_size = self._estimate_model_size(model)
        self.stats["original_size_bytes"] = original_size
        
        try:
            if self.target_precision in ["fp16", "float16"]:
                quantized_model = self._quantize_to_fp16(model)
            elif self.target_precision in ["bf16", "bfloat16"]:
                quantized_model = self._quantize_to_bf16(model)
            elif self.target_precision == "int8":
                quantized_model = self._quantize_to_int8(model)
            elif self.target_precision == "int4":
                quantized_model = self._quantize_to_int4(model)
            else:
                logger.warning(f"Unsupported precision {self.target_precision}, returning original model")
                return model
                
            # Calculate stats
            quantized_size = self._estimate_model_size(quantized_model)
            self.stats["quantized_size_bytes"] = quantized_size
            self.stats["size_reduction_percent"] = (1.0 - (quantized_size / original_size)) * 100
            self.stats["quantization_time_seconds"] = time.time() - start_time
            
            logger.info(f"Quantization complete. Size reduction: {self.stats['size_reduction_percent']:.2f}%")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Error during quantization: {str(e)}")
            return model
    
    def _quantize_to_fp16(self, model: 'torch.nn.Module') -> 'torch.nn.Module':
        """Quantize model to FP16 precision."""
        return model.half()
    
    def _quantize_to_bf16(self, model: 'torch.nn.Module') -> 'torch.nn.Module':
        """Quantize model to BF16 precision."""
        if hasattr(torch, 'bfloat16'):
            return model.to(torch.bfloat16)
        else:
            logger.warning("BF16 not supported in this PyTorch version, falling back to FP16")
            return model.half()
    
    def _quantize_to_int8(self, model: 'torch.nn.Module') -> 'torch.nn.Module':
        """Quantize model to INT8 precision."""
        try:
            import torch.quantization
            
            # Clone model for quantization
            qmodel = model
            
            # Ensure model is in eval mode
            qmodel.eval()
            
            if self.static_quantization:
                # Static quantization requires calibration
                # We need a forward function to collect statistics
                def collect_stats(self, x):
                    with torch.no_grad():
                        return self.forward(x)
                    
                # Set model to collect statistics during calibration
                qmodel.fuse_model = lambda: None  # Mock fuse function
                qmodel.qconfig = torch.quantization.get_default_qconfig('fbgemm' if torch.cuda.is_available() else 'qnnpack')
                torch.quantization.prepare(qmodel, inplace=True)
                
                # Run calibration (would need real data here)
                logger.warning("Skipping calibration - using dummy calibration data")
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)  # Example shape
                    _ = qmodel(dummy_input)
                
                # Convert to quantized model
                torch.quantization.convert(qmodel, inplace=True)
            else:
                # Dynamic quantization is simpler
                qmodel = torch.quantization.quantize_dynamic(
                    qmodel, 
                    {torch.nn.Linear, torch.nn.LSTM, torch.nn.LSTMCell, torch.nn.RNNCell, torch.nn.GRUCell},
                    dtype=torch.qint8
                )
                
            return qmodel
            
        except Exception as e:
            logger.error(f"INT8 quantization failed: {str(e)}")
            logger.warning("Falling back to FP16 quantization")
            return self._quantize_to_fp16(model)
    
    def _quantize_to_int4(self, model: 'torch.nn.Module') -> 'torch.nn.Module':
        """
        Quantize model to INT4 precision.
        
        Note: INT4 quantization is experimental and requires custom implementation
        as PyTorch doesn't natively support it.
        """
        try:
            # Check if we have optimum for quantization
            try:
                from optimum.gptq import GPTQQuantizer
                # This is just a placeholder - actual INT4 quantization would need calibration data
                logger.info("Using optimum.gptq for INT4 quantization")
                # Actual implementation would be more complex
                return model  # Placeholder - should return quantized model
            except ImportError:
                logger.warning("optimum.gptq not available, using custom INT4 quantization")
            
            # Custom INT4 quantization logic (simplified)
            # This is a naive implementation and would need to be expanded for real use
            
            # Create a copy of the model
            quantized_model = type(model)()
            quantized_model.load_state_dict(model.state_dict())
            
            # Dict to store quantized parameters
            int4_state_dict = {}
            
            # For each parameter, quantize to INT4
            with torch.no_grad():
                for name, param in model.state_dict().items():
                    if param.dtype == torch.float32 or param.dtype == torch.float16:
                        # Only quantize floating point tensors
                        if 'weight' in name and len(param.shape) > 1:
                            # Calculate scaling factor (per channel if enabled)
                            if self.per_channel and len(param.shape) > 1:
                                # Use per-output channel quantization for weight matrices
                                channel_dim = 0
                                max_abs_val, _ = torch.max(torch.abs(param), dim=1, keepdim=True)
                                scale = max_abs_val / 7.5  # INT4 range is -8 to 7
                                
                                # Quantize to INT4 precision
                                param_q = torch.round(param / scale).clamp(-8, 7)
                                
                                # Store in int8 format (PyTorch doesn't have int4)
                                # Will use two INT4 values per int8 byte when saving
                                param_q = param_q.to(torch.int8)
                                
                                # Pack two INT4 values per INT8 during saving (conceptual)
                                # In practice, would need to implement custom packing
                                
                                # Store quantized values and scale for dequantizing
                                int4_state_dict[name] = {'q_data': param_q, 'scale': scale}
                            else:
                                # Use a global scale
                                max_abs_val = torch.max(torch.abs(param))
                                scale = max_abs_val / 7.5
                                
                                param_q = torch.round(param / scale).clamp(-8, 7).to(torch.int8)
                                int4_state_dict[name] = {'q_data': param_q, 'scale': scale}
                        else:
                            # Don't quantize other parameters
                            int4_state_dict[name] = param
                    else:
                        # Non-float tensors are kept as-is
                        int4_state_dict[name] = param
            
            # Attach the quantized state dict to the model
            quantized_model.int4_state_dict = int4_state_dict
            
            # Override forward to use quantized weights
            original_forward = quantized_model.forward
            
            def quantized_forward(self, *args, **kwargs):
                # Save original state dict
                original_state_dict = {}
                with torch.no_grad():
                    for name, param in self.state_dict().items():
                        if name in self.int4_state_dict and isinstance(self.int4_state_dict[name], dict):
                            original_state_dict[name] = param.clone()
                            # Dequantize for inference
                            q_data = self.int4_state_dict[name]['q_data']
                            scale = self.int4_state_dict[name]['scale']
                            dequantized = q_data.float() * scale
                            param.copy_(dequantized)
                
                # Run inference with dequantized weights
                result = original_forward(*args, **kwargs)
                
                # Restore original state dict to save memory
                with torch.no_grad():
                    for name, param in original_state_dict.items():
                        self._parameters[name].copy_(param)
                
                return result
            
            # Bind the new forward method
            import types
            quantized_model.forward = types.MethodType(quantized_forward, quantized_model)
            
            logger.warning("INT4 quantization is experimental and may reduce model accuracy")
            return quantized_model
            
        except Exception as e:
            logger.error(f"INT4 quantization failed: {str(e)}")
            logger.warning("Falling back to INT8 quantization")
            return self._quantize_to_int8(model)
    
    def _estimate_model_size(self, model: 'torch.nn.Module') -> int:
        """Estimate the model size in bytes."""
        total_bytes = 0
        
        for param in model.parameters():
            # Calculate bytes per element based on dtype
            if param.dtype == torch.float32:
                bytes_per_element = 4
            elif param.dtype in [torch.float16, torch.bfloat16]:
                bytes_per_element = 2
            elif param.dtype == torch.int8:
                bytes_per_element = 1
            elif hasattr(torch, 'int4') and param.dtype == torch.int4:
                bytes_per_element = 0.5
            else:
                # Default for other types
                bytes_per_element = 4
            
            # Add parameter size
            total_bytes += param.numel() * bytes_per_element
        
        # Account for buffers (e.g., running stats in batch norm)
        for buffer in model.buffers():
            if buffer.dtype == torch.float32:
                bytes_per_element = 4
            elif buffer.dtype in [torch.float16, torch.bfloat16]:
                bytes_per_element = 2
            elif buffer.dtype == torch.int8:
                bytes_per_element = 1
            else:
                bytes_per_element = 4
            
            total_bytes += buffer.numel() * bytes_per_element
        
        return total_bytes
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get statistics about the quantization process."""

        return self.stats

    def _replace_forward_with_int4_matmul(self, 
                                        model: Any, 
                                        layer_name: str, 
                                        int4_weight: torch.Tensor, 
                                        scales: torch.Tensor, 
                                        zeros: Optional[torch.Tensor] = None):
        """
        Replace the forward method of a layer with a custom INT4 matmul.

        Args:
            model: Model containing the layer
            layer_name: Name of the layer in the model
            int4_weight: Packed INT4 weights
            scales: Scaling factors for quantization groups
            zeros: Zero points for asymmetric quantization (None for symmetric)
        """
        # Find the module by name
        module = model
        for attr in layer_name.split('.'):
            parent_module = module
            child_attr = attr
            module = getattr(module, attr)
        
        # Cache the original module for potential restoration
        orig_forward = module.forward
        
        # Define a new forward function with INT4 matmul
        def int4_forward(x):
            # Original bias if it exists
            bias = module.bias if hasattr(module, 'bias') and module.bias is not None else None
            
            # Get input shape and prepare for matmul
            orig_shape = x.shape
            x_reshaped = x.reshape(-1, x.shape[-1])
            
            # Unpack INT4 weights on-the-fly
            unpacked_weight = self._unpack_int4_weight(int4_weight, module.weight.shape, 
                                                      scales, zeros)
            
            # Perform matrix multiplication
            output = torch.matmul(x_reshaped, unpacked_weight.t())
            
            # Add bias if present
            if bias is not None:
                output = output + bias
                
            # Restore original batch dimensions
            output = output.reshape(orig_shape[:-1] + (output.shape[-1],))
            
            return output
        
        # Replace the forward method
        module.forward = int4_forward
        
        # Store original forward method for restoration
        self.quantized_modules[layer_name]["original_forward"] = orig_forward
    
    def _unpack_int4_weight(self, 
                           packed_weight: torch.Tensor, 
                           orig_shape: torch.Size,
                           scales: torch.Tensor, 
                           zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Unpack INT4 weights from the packed INT8 representation.
        
        Args:
            packed_weight: INT4 weights packed in INT8 tensor
            orig_shape: Original shape of weight tensor
            scales: Scaling factors for each group
            zeros: Zero points (None for symmetric)
            
        Returns:
            unpacked_weight: Unpacked and dequantized weights
        """
        # Determine number of elements in original weight
        num_elements = orig_shape[0] * orig_shape[1]
        num_packed = (num_elements + 1) // 2  # Ceiling division
        
        # Create tensor for unpacked int4 values
        unpacked_int4 = torch.zeros(packed_weight.shape[0], packed_weight.shape[1] * 2,
                                   dtype=torch.int8, device=packed_weight.device)
        
        # Unpack lower 4 bits
        unpacked_int4[:, ::2] = packed_weight & 0xF
        
        # Unpack upper 4 bits
        unpacked_int4[:, 1::2] = (packed_weight >> 4) & 0xF
        
        # Truncate any extra elements
        if unpacked_int4.shape[1] > orig_shape[1]:
            unpacked_int4 = unpacked_int4[:, :orig_shape[1]]
        
        # Reshape to match original weight shape
        unpacked_int4 = unpacked_int4.reshape(orig_shape)
        
        if zeros is None:
            # Symmetric: range is [-8, 7], stored as [-8, 7] directly
            # Convert from two's complement if negative
            unpacked_int4 = unpacked_int4.to(torch.int8)  # Ensure proper sign extension
            dequantized = unpacked_int4.to(torch.float32) * scales
        else:
            # Asymmetric: range is [0, 15], we need to subtract zero point
            dequantized = (unpacked_int4.to(torch.float32) - zeros) * scales
        
        return dequantized
    
    def restore_quantized_model(self, model: Any) -> Any:
        """
        Restore a quantized model back to its original weights.
        
        Args:
            model: Quantized model to restore
            
        Returns:
            restored_model: Model with original weights
        """
        for name, info in self.quantized_modules.items():
            if name == "weight_shape":
                continue
                
            try:
                # Find the module by name
                module = model
                for attr in name.split('.'):
                    module = getattr(module, attr)
                
                # Restore original weights
                if "original_weight" in info:
                    module.weight.data = info["original_weight"].clone()
                
                # Restore original forward method
                if "original_forward" in info:
                    module.forward = info["original_forward"]
                
                logger.info(f"Restored module {name} to original weights")
                
            except Exception as e:
                logger.error(f"Error restoring module {name}: {str(e)}")
        
        return model