from typing import Dict, Any, List, Union, Optional, Tuple
import logging
import os

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .base import ModelLoader
from ...runtime.core import DeviceType

logger = logging.getLogger(__name__)

class LLMLoader(ModelLoader):
    """Loader for Large Language Models using HuggingFace transformers."""
    
    def __init__(self, 
                 device_type: DeviceType = DeviceType.CUDA, 
                 device_id: int = 0,
                 quantization: Optional[str] = None):
        super().__init__(device_type, device_id)
        self.device = self._get_device()
        self.tokenizer = None
        self.quantization = quantization  # Options: None, "int8", "int4", "gptq"
        
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
        """Load a transformer LLM from local path or HuggingFace Hub."""
        try:
            # Configure loading options
            torch_dtype = kwargs.get("torch_dtype", torch.float16 if self.device.type != "cpu" else torch.float32)
            trust_remote_code = kwargs.get("trust_remote_code", False)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
            )
            
            # Load model with appropriate quantization if specified
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": trust_remote_code,
                "device_map": "auto" if self.device.type == "cuda" else None,
            }
            
            if self.quantization == "int8":
                load_kwargs["load_in_8bit"] = True
            elif self.quantization == "int4":
                load_kwargs["load_in_4bit"] = True
                load_kwargs["bnb_4bit_quant_type"] = "nf4"
                load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            # If not using device_map="auto", move model to device
            if self.device.type != "cuda" or "device_map" not in load_kwargs:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Store model configuration
            self.model_config = {
                "path": model_path,
                "device": str(self.device),
                "quantization": self.quantization,
                "model_type": self.model.config.model_type,
                **kwargs
            }
            
            logger.info(f"Loaded LLM model {model_path} to {self.device}")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            raise
    
    def get_inputs_info(self) -> List[Dict[str, Any]]:
        """Get information about model inputs."""
        if not self.verify_model():
            return []
        
        return [
            {"name": "input_ids", "shape": "dynamic", "dtype": "int64"},
            {"name": "attention_mask", "shape": "dynamic", "dtype": "int64"},
        ]
    
    def get_outputs_info(self) -> List[Dict[str, Any]]:
        """Get information about model outputs."""
        if not self.verify_model():
            return []
        
        return [
            {"name": "logits", "shape": "dynamic", "dtype": "float32"},
        ]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        if not self.verify_model():
            return {}
        
        config = self.model.config
        metadata = {
            "framework": "transformers",
            "model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)),
            "num_heads": getattr(config, "num_attention_heads", getattr(config, "n_head", 0)),
            "device": str(self.device),
            "quantization": self.quantization,
        }
        
        # Add info about model parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        metadata["num_parameters"] = num_params
        metadata["parameter_size_mb"] = num_params * 4 / (1024 * 1024)  # Approximation
        
        return metadata
    
    def infer(self, inputs: Union[str, List[str], Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Run inference on the given inputs (text or tensors)."""
        if not self.verify_model():
            raise ValueError("Model is not loaded")
            
        with torch.no_grad():
            # Process inputs
            if isinstance(inputs, (str, list)):
                # Encode text inputs
                encoded_inputs = self.tokenizer(
                    inputs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
            elif isinstance(inputs, dict):
                # Use provided tensor inputs
                encoded_inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")
            
            # Run inference
            outputs = self.model(**encoded_inputs)
            
            # Convert outputs to numpy
            return {
                "logits": outputs.logits.cpu().numpy(),
                "hidden_states": [h.cpu().numpy() for h in outputs.hidden_states] if outputs.hidden_states else None,
            }
    
    def generate(self, 
                prompt: Union[str, List[str]], 
                max_length: int = 100,
                temperature: float = 0.7,
                top_p: float = 0.9,
                top_k: int = 50,
                **kwargs) -> Dict[str, Any]:
        """Generate text using the language model."""
        if not self.verify_model():
            raise ValueError("Model is not loaded")
        
        # Encode the prompt
        input_ids = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).input_ids.to(self.device)
        
        # Set up generation parameters
        gen_kwargs = {
            "input_ids": input_ids,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)
        
        # Decode the generated text
        generated_texts = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        
        # Extract prompt and completion
        results = []
        for i, text in enumerate(generated_texts):
            if isinstance(prompt, list):
                prompt_text = prompt[i]
            else:
                prompt_text = prompt
                
            completion = text[len(self.tokenizer.decode(input_ids[i], skip_special_tokens=True)):]
            
            results.append({
                "prompt": prompt_text,
                "completion": completion,
                "full_text": text
            })
        
        return results[0] if not isinstance(prompt, list) else results