"""
OpenInference: High-Performance AI Inference Engine

This module provides the main integration point for the OpenInference system.
"""

import logging
import sys
import os
from typing import Dict, Any, Optional, List
import queue

from .hardware.accelerator import HardwareManager, AcceleratorType
from .models.registry import ModelRegistry
from .runtime.continuous_batching import ContinuousBatcher as RuntimeContinuousBatcher
from .monitoring.performance_tracker import PerformanceTracker
from .runtime.memory_manager import MemoryManager, KVCacheManager
from .optimization.quantization import PyTorchQuantizer
from .api.server import start_server

logger = logging.getLogger(__name__)

class OpenInference:
    """
    Main class for the OpenInference system.
    
    Provides a unified interface for model loading, optimization,
    inference execution, and performance monitoring.
    """
    
    def __init__(self,
                device: Optional[str] = None,
                models_dir: str = "models",
                cache_dir: str = ".cache",
                max_batch_size: int = 32,
                enable_continuous_batching: bool = True):
        """
        Initialize the OpenInference system.
        
        Args:
            device: Preferred device type (e.g., "cuda", "cpu", "metal")
            models_dir: Directory containing models
            cache_dir: Directory for caching models
            max_batch_size: Maximum batch size for inference
            enable_continuous_batching: Whether to enable continuous batching for LLMs
        """
        # Initialize hardware manager
        device_type = AcceleratorType(device) if device else None
        self.hardware_manager = HardwareManager(prefer_device_type=device_type)
        
        # Initialize model registry
        self.model_registry = ModelRegistry(
            models_dir=models_dir,
            cache_dir=cache_dir,
            hardware_manager=self.hardware_manager
        )
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            device=self.hardware_manager.get_device_str()
        )
        
        # Initialize KV cache manager for transformer models
        self.kv_cache_manager = KVCacheManager(
            device=self.hardware_manager.get_device_str(),
            max_memory_fraction=0.8  # Use up to 80% of GPU memory for KV cache
        )
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(
            record_gpu_metrics=True,
            export_metrics=True
        )
        
        # Batching settings
        self.max_batch_size = max_batch_size
        self.enable_continuous_batching = enable_continuous_batching
        
        # Initialize batchers
        self.dynamic_batcher = DynamicBatcher(
            max_batch_size=max_batch_size,
            max_wait_time_ms=100,  # Max time to wait for batch completion
            performance_tracker=self.performance_tracker
        )
        
        # Continuous batcher for LLMs (will be initialized per model)
        self.continuous_batchers: Dict[str, RuntimeContinuousBatcher] = {}
        
        # Loaded models
        self.loaded_models = {}
        
        logger.info("OpenInference system initialized")
        logger.info(f"Selected device: {self.hardware_manager.get_device_str()}")
    
    def load_model(self, model_name: str, quantize: Optional[str] = None) -> bool:
        """
        Load a model for inference.
        
        Args:
            model_name: Name of the model to load
            quantize: Quantization level (e.g., "int8", "fp16")
            
        Returns:
            success: Whether the model was loaded successfully
        """
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Load the model from registry
            model = self.model_registry.load_model(model_name)
            
            if model is None:
                logger.error(f"Failed to load model: {model_name}")
                return False
            
            # Apply hardware optimizations
            model = self.hardware_manager.optimize_model_for_device(model)
            
            # Apply quantization if requested
            if quantize:
                logger.info(f"Applying {quantize} quantization to model")
                quantizer = PyTorchQuantizer(target_precision=quantize)
                model = quantizer.quantize(model)
            
            # Add to loaded models
            self.loaded_models[model_name] = {
                "model": model,
                "config": self.model_registry.get_model_config(model_name),
                "tokenizer": self.model_registry.get_tokenizer(model_name)
            }
            
            # Initialize continuous batcher for transformer models if enabled
            if self.enable_continuous_batching and self._is_transformer_model(model_name):
                tokenizer = self.loaded_models[model_name]["tokenizer"]
                
                if tokenizer is not None:
                    self.continuous_batchers[model_name] = RuntimeContinuousBatcher(
                        model=model,
                        device=self.hardware_manager.get_device_str(),
                        max_batch_size=self.max_batch_size,
                        max_input_length=self.loaded_models[model_name]["config"].max_position_embeddings if hasattr(self.loaded_models[model_name]["config"], 'max_position_embeddings') else 2048,
                        max_prefill_tokens=4096,
                        max_attention_window=self.loaded_models[model_name]["config"].max_position_embeddings if hasattr(self.loaded_models[model_name]["config"], 'max_position_embeddings') else 2048,
                    )
                    
                    # Start the continuous batcher
                    self.continuous_batchers[model_name].start()
            
            logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            success: Whether the model was unloaded successfully
        """
        try:
            if model_name not in self.loaded_models:
                logger.warning(f"Model {model_name} not loaded")
                return False
            
            # Stop continuous batcher if exists
            if model_name in self.continuous_batchers:
                self.continuous_batchers[model_name].stop()
                del self.continuous_batchers[model_name]
            
            # Remove model from loaded models
            del self.loaded_models[model_name]
            
            # Explicitly trigger garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if hasattr(self.memory_manager, 'clear_cache'):
                self.memory_manager.clear_cache()
            
            logger.info(f"Model {model_name} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {str(e)}")
            return False
    
    def _is_transformer_model(self, model_name: str) -> bool:
        """Check if a model is a transformer-based model."""
        if model_name not in self.loaded_models:
            return False
            
        config = self.loaded_models[model_name]["config"]
        
        # Check for common transformer attributes
        if hasattr(config, 'model_type'):
            transformer_types = [
                'gpt', 'llama', 'bert', 'roberta', 't5', 'bart', 
                'bloom', 'gpt_neox', 'gptj', 'opt', 'falcon'
            ]
            return any(t in config.model_type.lower() for t in transformer_types)
        
        # Check for architecture-specific attributes
        return (hasattr(config, 'hidden_size') and
                (hasattr(config, 'num_hidden_layers') or hasattr(config, 'n_layer')) and
                (hasattr(config, 'num_attention_heads') or hasattr(config, 'n_head')))
    
    def run_inference(self, 
                     model_name: str, 
                     inputs: Any,
                     batch_size: Optional[int] = None,
                     stream: bool = False,
                     **kwargs) -> Any:
        """
        Run inference with a model.
        
        Args:
            model_name: Name of the loaded model
            inputs: Input data for inference
            batch_size: Batch size override
            stream: Whether to stream results (for LLMs)
            **kwargs: Additional model-specific parameters
            
        Returns:
            outputs: Model outputs
        """
        if model_name not in self.loaded_models:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        # Start tracking this request
        request_id = f"req_{model_name}_{str(id(inputs))[-8:]}"
        tracking_info = self.performance_tracker.start_request(
            request_id=request_id,
            model_name=model_name
        )
        
        try:
            # For transformer models with continuous batching and streaming
            if stream and model_name in self.continuous_batchers:
                tokenizer = self.loaded_models[model_name].get("tokenizer")
                if not tokenizer:
                    logger.error(f"Tokenizer not found for model {model_name}, cannot stream.")
                    self.performance_tracker.finish_request(tracking_info=tracking_info, success=False)
                    return None

                try:
                    prompt_token_ids = tokenizer.encode(inputs, add_special_tokens=True)
                except Exception as e:
                    logger.error(f"Error tokenizing input for model {model_name}: {str(e)}")
                    self.performance_tracker.finish_request(tracking_info=tracking_info, success=False)
                    return None

                token_queue = queue.Queue()
                
                # Define the callback for the runtime batcher (receives token IDs)
                def runtime_batcher_callback(generated_token_ids: List[int], is_done: bool):
                    # This callback is from the batcher's thread.
                    # It should put data onto the queue for the main thread's generator.
                    token_queue.put((generated_token_ids, is_done))

                # Submit generation request
                # kwargs might include: max_new_tokens, temperature, top_p, top_k, do_sample
                max_new_tokens = kwargs.get("max_new_tokens", self.loaded_models[model_name]["config"].max_length if hasattr(self.loaded_models[model_name]["config"], 'max_length') else 256)
                
                self.continuous_batchers[model_name].submit_request(
                    prompt_tokens=prompt_token_ids,
                    max_new_tokens=max_new_tokens,
                    callback=runtime_batcher_callback,
                    **kwargs # Pass other generation params like temp, top_p, etc.
                )
                
                # For streaming, return a generator
                def result_generator():
                    all_generated_ids = []
                    try:
                        while True:
                            # Get token IDs with a timeout
                            chunk_token_ids, is_done = token_queue.get(timeout=kwargs.get("timeout", 120.0))
                            
                            if chunk_token_ids: # It seems runtime batcher sends list of new tokens
                                all_generated_ids.extend(chunk_token_ids)
                                # Decode only the new chunk to yield
                                new_text_chunk = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
                                if new_text_chunk: # Avoid yielding empty strings if decode results in that
                                     yield new_text_chunk
                            
                            if is_done:
                                break
                    except queue.Empty:
                        logger.warning(f"Timeout waiting for token stream from runtime batcher for request {request_id}")
                    except Exception as e:
                        logger.error(f"Error in stream generator for request {request_id} with runtime batcher: {e}", exc_info=True)
                    finally:
                        # Ensure performance tracking is finished
                        self.performance_tracker.finish_request(
                            tracking_info=tracking_info,
                            success=True # TODO: Add more nuanced success tracking if possible
                        )
                
                return result_generator()
            
            # For standard models using dynamic batching or non-streaming LLM inference
            else:
                # Get model and required components
                model = self.loaded_models[model_name]["model"]
                
                # Prepare compute function
                def compute_fn(batch_inputs):
                    # Record computation start time
                    compute_start = time.time()
                    
                    # Move inputs to device if needed
                    if hasattr(batch_inputs, 'to') and hasattr(model, 'device'):
                        batch_inputs = batch_inputs.to(model.device)
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(batch_inputs, **kwargs)
                    
                    # Record completion
                    input_shape = list(batch_inputs.shape) if hasattr(batch_inputs, 'shape') else None
                    output_shape = list(outputs.shape) if hasattr(outputs, 'shape') else None
                    
                    self.performance_tracker.finish_request(
                        tracking_info=tracking_info,
                        batch_size=batch_inputs.shape[0] if hasattr(batch_inputs, 'shape') and len(batch_inputs.shape) > 0 else 1,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        compute_start_time=compute_start,
                        success=True
                    )
                    
                    return outputs
                
                # Use dynamic batcher if enabled and batching makes sense
                if batch_size and batch_size > 1:
                    return self.dynamic_batcher.process(
                        inputs=inputs,
                        compute_fn=compute_fn,
                        batch_size=batch_size
                    )
                else:
                    # Direct computation without batching
                    return compute_fn(inputs)
                
        except Exception as e:
            logger.error(f"Inference error with model {model_name}: {str(e)}")
            
            # Record failure
            self.performance_tracker.finish_request(
                tracking_info=tracking_info,
                success=False
            )
            
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        metrics = self.performance_tracker.get_metrics_summary()
        
        # Add hardware utilization
        device_info = self.hardware_manager.get_selected_device_info()
        if device_info:
            metrics["hardware"] = device_info.to_dict()
        
        # Add memory usage
        metrics["memory"] = self.memory_manager.get_memory_stats()
        
        # Add KV cache stats if available
        if hasattr(self.kv_cache_manager, 'get_stats'):
            metrics["kv_cache"] = self.kv_cache_manager.get_stats()
        
        # Add batcher stats
        metrics["dynamic_batcher"] = self.dynamic_batcher.get_stats()
        
        # Add continuous batcher stats if any
        if self.continuous_batchers:
            metrics["continuous_batchers"] = {
                model_name: batcher.get_stats()
                for model_name, batcher in self.continuous_batchers.items()
            }
        
        return metrics
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 4):
        """
        Start REST API server for inference.
        
        Args:
            host: Hostname to bind
            port: Port to bind
            workers: Number of worker processes
        """
        start_server(
            host=host,
            port=port,
            workers=workers,
            model_registry=self.model_registry,
            hardware_manager=self.hardware_manager,
            performance_tracker=self.performance_tracker,
            inference_engine=self
        )
    
    def shutdown(self):
        """Shutdown the inference system."""
        # Stop all continuous batchers
        for model_name, batcher in list(self.continuous_batchers.items()):
            batcher.stop()
        
        # Unload all models
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        
        # Stop performance tracker
        self.performance_tracker.stop()
        
        # Clear memory
        self.memory_manager.clear_all()
        self.kv_cache_manager.clear_all()
        
        logger.info("OpenInference system shutdown complete")