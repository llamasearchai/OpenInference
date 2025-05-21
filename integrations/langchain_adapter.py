"""
LangChain adapter for OpenInference.

This module provides the necessary classes and functions to integrate
OpenInference with LangChain, allowing OpenInference to be used as a custom LLM
within LangChain workflows.
"""

from typing import Any, Dict, List, Mapping, Optional, Union
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

from .. import OpenInference

class OpenInferenceLLM(LLM):
    """
    LangChain integration for OpenInference.
    
    This class allows OpenInference models to be used as LLMs within LangChain.
    """
    
    model_name: str
    """Name of the model to use."""
    
    inference_engine: Optional[OpenInference] = None
    """Optional pre-initialized inference engine."""
    
    streaming: bool = False
    """Whether to stream the results."""
    
    max_new_tokens: int = 256
    """Maximum number of new tokens to generate."""
    
    temperature: float = 0.7
    """Temperature for sampling."""
    
    top_p: float = 0.9
    """Top-p sampling parameter."""
    
    top_k: int = 50
    """Top-k sampling parameter."""
    
    def __init__(self, **kwargs):
        """Initialize the OpenInferenceLLM."""
        super().__init__(**kwargs)
        
        # Initialize engine if not provided
        if self.inference_engine is None:
            self.inference_engine = OpenInference()
            
        # Ensure model is loaded
        if not self.inference_engine.model_registry.is_model_loaded(self.model_name):
            self.inference_engine.load_model(self.model_name)
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "openinference"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Run the LLM on the given prompt."""
        # Merge instance params with call params
        params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        params.update(kwargs)
        
        # Add stop tokens if provided
        if stop:
            params["stop_sequences"] = stop
        
        # Handle streaming and non-streaming cases
        if self.streaming and run_manager:
            # For streaming
            text_callback = run_manager.on_llm_new_token
            
            full_response = ""
            for chunk in self.inference_engine.run_inference(
                self.model_name, prompt, stream=True, **params
            ):
                text_callback(chunk)
                full_response += chunk
            
            return full_response
        else:
            # For non-streaming
            return self.inference_engine.run_inference(
                self.model_name, prompt, stream=False, **params
            )

print("Langchain adapter placeholder created. Further implementation needed.") 