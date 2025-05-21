from typing import Dict, Any, List, Optional, Union
import logging
import time
import os
import json
from enum import Enum

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..models.loaders import PyTorchModelLoader, LLMLoader
from ..batching import DynamicBatcher, ContinuousBatcher
from ..monitoring import PerformanceTracker

logger = logging.getLogger(__name__)

# Define API models
class ModelType(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    LLM = "llm"

class LoadModelRequest(BaseModel):
    model_path: str
    model_type: ModelType
    device: str = "cuda"
    device_id: int = 0
    quantization: Optional[str] = None
    batch_size: int = 16
    max_sequence_length: int = 2048
    extra_config: Dict[str, Any] = Field(default_factory=dict)

class InferenceRequest(BaseModel):
    model_id: str
    inputs: Union[List[float], List[List[float]], Dict[str, Any]]
    input_shape: Optional[List[int]] = None

class TextGenerationRequest(BaseModel):
    model_id: str
    prompt: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False
    extra_params: Dict[str, Any] = Field(default_factory=dict)

class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    device: str
    metadata: Dict[str, Any]

# Create FastAPI app
app = FastAPI(
    title="OpenInference API",
    description="High-Performance AI Inference Engine API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Customize in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
loaded_models = {}  # model_id -> model_loader
batchers = {}       # model_id -> batcher
model_configs = {}  # model_id -> config

# Create performance tracker
performance_tracker = PerformanceTracker(
    max_history=1000,
    export_path=os.environ.get("METRICS_EXPORT_PATH"),
    record_gpu_metrics=True
)

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    performance_tracker.start()
    logger.info("OpenInference API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    performance_tracker.stop()
    
    # Stop all batchers
    for batcher in batchers.values():
        batcher.stop()
    
    logger.info("OpenInference API shutting down")

@app.post("/v1/models/load", response_model=ModelInfo)
async def load_model(request: LoadModelRequest):
    """Load a model and prepare it for inference."""
    model_id = f"{request.model_type}_{int(time.time())}"
    
    try:
        # Create appropriate model loader based on type
        if request.model_type == ModelType.LLM:
            device_type = "cuda" if request.device == "cuda" else "cpu"
            loader = LLMLoader(
                device_type=device_type,
                device_id=request.device_id,
                quantization=request.quantization
            )
        elif request.model_type == ModelType.PYTORCH:
            device_type = "cuda" if request.device == "cuda" else "cpu"
            loader = PyTorchModelLoader(
                device_type=device_type,
                device_id=request.device_id
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {request.model_type}")
        
        # Load the model
        loader.load(request.model_path, **request.extra_config)
        
        # Store in global state
        loaded_models[model_id] = loader
        model_configs[model_id] = request.dict()
        
        # Create appropriate batcher
        if request.model_type == ModelType.LLM:
            batcher = ContinuousBatcher(
                model_fn=loader.model.forward,
                tokenizer=loader.tokenizer,
                max_batch_size=request.batch_size,
                max_sequence_length=request.max_sequence_length,
                device=request.device
            )
        else:
            # Create a dynamic batcher for regular models
            batcher = DynamicBatcher(
                process_fn=lambda inputs: [loader.infer(x) for x in inputs],
                max_batch_size=request.batch_size,
                max_wait_time=0.01
            )
        
        batcher.start()
        batchers[model_id] = batcher
        
        # Get model metadata
        metadata = loader.get_metadata()
        
        return ModelInfo(
            model_id=model_id,
            model_type=request.model_type,
            device=request.device,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/v1/models/unload")
async def unload_model(model_id: str):
    """Unload a model and free resources."""
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    try:
        # Stop the batcher if exists
        if model_id in batchers:
            batchers[model_id].stop()
            del batchers[model_id]
        
        # Remove the model
        del loaded_models[model_id]
        del model_configs[model_id]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"success": True, "message": f"Model {model_id} unloaded"}
        
    except Exception as e:
        logger.error(f"Failed to unload model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

@app.post("/v1/models/list")
async def list_models():
    """List all loaded models."""
    model_list = []
    for model_id, loader in loaded_models.items():
        model_list.append({
            "model_id": model_id,
            "model_type": model_configs[model_id]["model_type"],
            "device": model_configs[model_id]["device"],
            "metadata": loader.get_metadata()
        })
    
    return {"models": model_list}

@app.post("/v1/inference")
async def run_inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    """Run inference on a loaded model."""
    if request.model_id not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    
    loader = loaded_models[request.model_id]
    batcher = batchers[request.model_id]
    
    # Start timing
    tracking_info = performance_tracker.start_request(
        request_id=f"inf_{int(time.time())}",
        model_name=request.model_id
    )
    
    compute_start_time = time.time()
    
    try:
        # Prepare inputs
        inputs = request.inputs
        
        # Wait for result with timeout
        from concurrent.futures import Future
        result_future = Future()
        
        def on_complete(result, success):
            if success:
                result_future.set_result(result)
            else:
                result_future.set_exception(Exception("Inference failed"))
        
        # Submit to batcher
        batcher.submit(inputs, on_complete)
        
        # Wait for result with timeout (30 seconds)
        import asyncio
        try:
            result = await asyncio.wait_for(asyncio.wrap_future(result_future), timeout=30.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Inference timed out")
        
        # Record performance metrics
        input_shape = request.input_shape or [1]
        output_shape = [1]  # Placeholder
        if isinstance(result, np.ndarray):
            output_shape = list(result.shape)
        
        # Use background task to avoid blocking response
        background_tasks.add_task(
            performance_tracker.finish_request,
            tracking_info=tracking_info,
            batch_size=1,
            input_shape=input_shape,
            output_shape=output_shape,
            compute_start_time=compute_start_time,
            success=True
        )
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(result, np.ndarray):
            result = result.tolist()
        elif isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    result[k] = v.tolist()
        
        return {"result": result}
        
    except Exception as e:
        # Record failure
        background_tasks.add_task(
            performance_tracker.finish_request,
            tracking_info=tracking_info,
            batch_size=1,
            input_shape=request.input_shape or [1],
            output_shape=[1],
            compute_start_time=compute_start_time,
            success=False,
            error_message=str(e)
        )
        
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/v1/generate")
async def generate_text(request: TextGenerationRequest, background_tasks: BackgroundTasks):
    """Generate text using a loaded language model."""
    if request.model_id not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    
    # Validate model type
    model_type = model_configs[request.model_id]["model_type"]
    if model_type != ModelType.LLM:
        raise HTTPException(status_code=400, detail=f"Model {request.model_id} is not a language model")
    
    loader = loaded_models[request.model_id]
    batcher = batchers[request.model_id]
    
    # Start timing
    tracking_info = performance_tracker.start_request(
        request_id=f"gen_{int(time.time())}",
        model_name=request.model_id
    )
    
    compute_start_time = time.time()
    
    try:
        # Set up generation parameters
        gen_params = {
            "max_length": request.max_length,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            **request.extra_params
        }
        
        if request.stream:
            # Streaming generation (SSE)
            async def generate_stream():
                # First yield headers
                yield "data: " + json.dumps({"status": "starting"}) + "\n\n"
                
                # Set up streaming
                from asyncio import Queue
                queue = Queue()
                
                # Function to handle token streaming
                def on_token(token, finished):
                    queue.put_nowait((token, finished))
                
                # Start generation
                batcher.submit(request.prompt, on_token, **gen_params)
                
                # Stream tokens as they arrive
                text_so_far = ""
                while True:
                    # Wait for next token
                    token, finished = await queue.get()
                    text_so_far += token
                    
                    # Yield the token
                    yield "data: " + json.dumps({
                        "text": text_so_far,
                        "token": token,
                        "finished": finished
                    }) + "\n\n"
                    
                    # If generation is finished, exit
                    if finished:
                        # Record performance metrics
                        background_tasks.add_task(
                            performance_tracker.finish_request,
                            tracking_info=tracking_info,
                            batch_size=1,
                            input_shape=[len(request.prompt)],
                            output_shape=[len(text_so_far)],
                            compute_start_time=compute_start_time,
                            success=True,
                            extra_data={"tokens_generated": len(text_so_far)}
                        )
                        break
            
            # Return streaming response
            return Response(
                content=generate_stream(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming generation
            from concurrent.futures import Future
            result_future = Future()
            
            def on_complete(result, success):
                if success:
                    result_future.set_result(result)
                else:
                    result_future.set_exception(Exception("Generation failed"))
            
            # Submit to batcher
            batcher.submit(request.prompt, on_complete, **gen_params)
            
            # Wait for result with timeout
            import asyncio
            try:
                result = await asyncio.wait_for(asyncio.wrap_future(result_future), timeout=60.0)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="Text generation timed out")
            
            # Record performance metrics
            background_tasks.add_task(
                performance_tracker.finish_request,
                tracking_info=tracking_info,
                batch_size=1,
                input_shape=[len(request.prompt)],
                output_shape=[len(result)],
                compute_start_time=compute_start_time,
                success=True,
                extra_data={"tokens_generated": len(result)}
            )
            
            return {"generated_text": result}
            
    except Exception as e:
        # Record failure
        background_tasks.add_task(
            performance_tracker.finish_request,
            tracking_info=tracking_info,
            batch_size=1,
            input_shape=[len(request.prompt)],
            output_shape=[0],
            compute_start_time=compute_start_time,
            success=False,
            error_message=str(e)
        )
        
        logger.error(f"Text generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.get("/v1/metrics")
async def get_metrics():
    """Get performance metrics."""
    return {
        "stats": performance_tracker.get_stats(),
        "recent_requests": performance_tracker.get_recent_metrics(20)
    }

def start_server(host="0.0.0.0", port=8000, log_level="info"):
    """Start the FastAPI server."""
    uvicorn.run("openinference.api.server:app", host=host, port=port, log_level=log_level)

if __name__ == "__main__":
    start_server()