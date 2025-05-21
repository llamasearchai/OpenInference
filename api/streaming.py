"""
Token streaming utilities for the API layer.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator, Union

logger = logging.getLogger(__name__)

class TokenStreamingManager:
    """
    Manages streaming of tokens to clients via API endpoints.
    
    Provides utilities for converting token IDs to text,
    handling streamed responses, and managing stream timeouts.
    """
    
    def __init__(self, tokenizer: Any, chunk_size: int = 1):
        """
        Initialize the token streaming manager.
        
        Args:
            tokenizer: Tokenizer for converting IDs to text
            chunk_size: Number of tokens to group per chunk (1 for immediate streaming)
        """
        self.tokenizer = tokenizer
        self.chunk_size = max(1, chunk_size)
        self.active_streams = {}
        
        # Track usage statistics
        self.stats = {
            "total_streams": 0,
            "active_streams": 0,
            "total_tokens_streamed": 0
        }
    
    def create_token_processor(self, 
                              stream_id: str, 
                              skip_special_tokens: bool = True,
                              skip_prompt: bool = False,
                              include_metadata: bool = True) -> Callable[[List[int], bool], None]:
        """
        Create a token processor function for handling streamed tokens.
        
        Args:
            stream_id: Unique ID for this stream
            skip_special_tokens: Whether to skip special tokens
            skip_prompt: Whether to skip prompt tokens
            include_metadata: Whether to include metadata in streamed response
            
        Returns:
            processor: Function to process new tokens
        """
        # Setup stream state
        buffer = []
        output_tokens = []
        queue = asyncio.Queue()
        start_time = time.time()
        first_token_time = None
        
        # Register stream
        self.active_streams[stream_id] = {
            "queue": queue,
            "start_time": start_time,
            "first_token_time": None,
            "tokens_generated": 0,
            "last_activity": start_time
        }
        
        # Update stats
        self.stats["total_streams"] += 1
        self.stats["active_streams"] += 1
        
        def process_token(tokens: List[int], is_done: bool) -> None:
            """Process new tokens and optionally send to client."""
            nonlocal first_token_time, buffer, output_tokens
            
            # Update stream state
            stream_state = self.active_streams.get(stream_id)
            if not stream_state:
                return
            
            # Record first token time
            if tokens and stream_state["first_token_time"] is None:
                stream_state["first_token_time"] = time.time()
                first_token_time = stream_state["first_token_time"]
            
            # Update last activity
            stream_state["last_activity"] = time.time()
            
            # Add tokens to buffer
            buffer.extend(tokens)
            output_tokens.extend(tokens)
            
            # Update stats
            stream_state["tokens_generated"] += len(tokens)
            self.stats["total_tokens_streamed"] += len(tokens)
            
            # Decide whether to flush based on chunk size or stream ending
            if len(buffer) >= self.chunk_size or is_done:
                token_text = self.tokenizer.decode(
                    buffer, skip_special_tokens=skip_special_tokens
                )
                
                # Prepare chunk data
                chunk = {
                    "text": token_text,
                    "tokens": len(buffer)
                }
                
                # Add metadata if requested
                if include_metadata:
                    elapsed = time.time() - start_time
                    time_to_first = None
                    if first_token_time:
                        time_to_first = first_token_time - start_time
                    
                    chunk["metadata"] = {
                        "id": stream_id,
                        "tokens_generated": stream_state["tokens_generated"],
                        "elapsed_time": elapsed,
                        "time_to_first_token": time_to_first,
                        "tokens_per_second": stream_state["tokens_generated"] / max(0.001, elapsed)
                    }
                
                # Add completion flag
                if is_done:
                    chunk["done"] = True
                    
                    # For the final chunk, include the full text if requested
                    if include_metadata:
                        chunk["metadata"]["complete_text"] = self.tokenizer.decode(
                            output_tokens, skip_special_tokens=skip_special_tokens
                        )
                
                # Add to queue for async processing
                try:
                    queue.put_nowait(chunk)
                except Exception as e:
                    logger.error(f"Error adding to stream queue {stream_id}: {str(e)}")
                
                # Clear buffer after sending
                buffer = []
            
            # Clean up if done
            if is_done:
                # Remove from active streams
                self.active_streams.pop(stream_id, None)
                self.stats["active_streams"] -= 1
                
                # Signal end of stream
                try:
                    queue.put_nowait(None)
                except Exception as e:
                    logger.error(f"Error signaling end of stream {stream_id}: {str(e)}")
        
        return process_token
    
    async def stream_generator(self, stream_id: str) -> AsyncGenerator[str, None]:
        """
        Create an async generator for streaming tokens to clients.
        
        Args:
            stream_id: Unique ID for this stream
            
        Yields:
            chunk: JSON string containing token text and metadata
        """
        # Get stream state
        stream_state = self.active_streams.get(stream_id)
        if not stream_state:
            logger.warning(f"Stream {stream_id} not found")
            return
        
        queue = stream_state["queue"]
        
        # Stream until explicitly ended
        while True:
            try:
                # Wait for new chunks with timeout
                chunk = await asyncio.wait_for(queue.get(), timeout=60.0)
                
                # None signals end of stream
                if chunk is None:
                    break
                
                # Yield JSON-encoded chunk
                yield json.dumps(chunk) + "\n"
                
            except asyncio.TimeoutError:
                # Check if stream is still active
                if stream_id not in self.active_streams:
                    break
                
                # Check for inactivity timeout
                current_time = time.time()
                last_activity = stream_state.get("last_activity", 0)
                
                if current_time - last_activity > 60.0:  # 60 second inactivity timeout
                    logger.warning(f"Stream {stream_id} timed out due to inactivity")
                    # Yield timeout message
                    yield json.dumps({
                        "error": "Stream timeout due to inactivity",
                        "done": True
                    }) + "\n"
                    break
                
                # Otherwise just continue waiting
                continue
                
            except Exception as e:
                logger.error(f"Error in stream generator for {stream_id}: {str(e)}")
                # Yield error message
                yield json.dumps({
                    "error": f"Stream error: {str(e)}",
                    "done": True
                }) + "\n"
                break
        
        # Clean up stream if still active
        if stream_id in self.active_streams:
            self.active_streams.pop(stream_id)
            self.stats["active_streams"] -= 1
    
    def cancel_stream(self, stream_id: str) -> bool:
        """
        Cancel an active token stream.
        
        Args:
            stream_id: Stream ID to cancel
            
        Returns:
            success: Whether cancellation was successful
        """
        if stream_id not in self.active_streams:
            logger.warning(f"Cannot cancel non-existent stream {stream_id}")
            return False
        
        # Get stream state
        stream_state = self.active_streams.get(stream_id)
        
        try:
            # Signal end of stream
            stream_state["queue"].put_nowait(None)
            
            # Remove from active streams
            self.active_streams.pop(stream_id, None)
            self.stats["active_streams"] -= 1
            
            logger.debug(f"Cancelled stream {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling stream {stream_id}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about token streaming."""
        current_stats = dict(self.stats)
        current_stats["current_active_streams"] = len(self.active_streams)
        
        # Calculate average tokens per second across active streams
        if self.active_streams:
            total_tps = 0
            for stream_id, state in self.active_streams.items():
                elapsed = time.time() - state["start_time"]
                if elapsed > 0:
                    total_tps += state["tokens_generated"] / elapsed
            
            current_stats["avg_tokens_per_second"] = total_tps / len(self.active_streams)
        else:
            current_stats["avg_tokens_per_second"] = 0
            
        return current_stats