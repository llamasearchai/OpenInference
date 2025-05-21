from typing import Callable, Any, List, Dict, Optional, Union, Iterator
import asyncio
import queue
import threading
import time

class TokenStreamer:
    """
    Utility for streaming tokens from a language model to a callback.
    
    This allows clients to receive tokens as they're generated rather than
    waiting for the full response.
    """
    
    def __init__(self):
        """Initialize the token streamer."""
        self.active_streams = {}
    
    def create_stream(self, stream_id: str) -> 'TokenStream':
        """Create a new token stream."""
        stream = TokenStream(stream_id, self._on_stream_finished)
        self.active_streams[stream_id] = stream
        return stream
    
    def _on_stream_finished(self, stream_id: str):
        """Called when a stream is finished."""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
    
    def get_active_stream_count(self) -> int:
        """Get the number of active streams."""
        return len(self.active_streams)

class TokenStream:
    """
    A stream of tokens for a specific inference request.
    
    Provides methods to add tokens to the stream and ways for consumers
    to receive tokens as they become available.
    """
    
    def __init__(self, stream_id: str, on_finished_callback: Callable[[str], None]):
        """Initialize the token stream."""
        self.stream_id = stream_id
        self.on_finished_callback = on_finished_callback
        self.is_finished = False
        self.error = None
        
        # For synchronous consumers
        self.token_queue = queue.Queue()
        
        # For async consumers
        self.async_queue = asyncio.Queue()
        
        # For callback-based consumers
        self.callbacks = []
        
        # Store all tokens
        self.all_tokens = []
    
    def add_token(self, token: str) -> None:
        """Add a token to the stream."""
        if self.is_finished:
            return
            
        self.all_tokens.append(token)
        
        # Add to queues
        self.token_queue.put(token)
        
        # Add to async queue if running in async context
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: asyncio.create_task(self.async_queue.put(token))
            )
        except (RuntimeError, AttributeError):
            # Not in an async context or event loop not running
            pass
        
        # Call callbacks
        for callback in self.callbacks:
            try:
                callback(token, False)
            except Exception as e:
                print(f"Error in token callback: {str(e)}")
    
    def finish(self, error: Optional[str] = None) -> None:
        """Mark the stream as finished."""
        if self.is_finished:
            return
            
        self.is_finished = True
        self.error = error
        
        # Signal queue consumers
        self.token_queue.put(None)
        
        # Signal async queue consumers
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: asyncio.create_task(self.async_queue.put(None))
            )
        except (RuntimeError, AttributeError):
            # Not in an async context or event loop not running
            pass
        
        # Signal callback consumers
        for callback in self.callbacks:
            try:
                callback("", True)
            except Exception as e:
                print(f"Error in token callback: {str(e)}")
        
        # Call finished callback
        if self.on_finished_callback:
            try:
                self.on_finished_callback(self.stream_id)
            except Exception as e:
                print(f"Error in finished callback: {str(e)}")
    
    def add_callback(self, callback: Callable[[str, bool], None]) -> None:
        """
        Add a callback to receive tokens.
        
        The callback will be called with (token, is_finished) arguments.
        When is_finished is True, token will be empty.
        """
        self.callbacks.append(callback)
        
        # Send all existing tokens
        for token in self.all_tokens:
            try:
                callback(token, False)
            except Exception as e:
                print(f"Error in token callback: {str(e)}")
        
        # If already finished, send finished signal
        if self.is_finished:
            try:
                callback("", True)
            except Exception as e:
                print(f"Error in token callback: {str(e)}")
    
    def iter_tokens(self) -> Iterator[str]:
        """
        Iterate over tokens as they become available.
        
        This is a blocking iterator that yields tokens until the stream is finished.
        """
        while True:
            token = self.token_queue.get()
            if token is None:  # Stream finished
                break
            yield token
    
    async def aiter_tokens(self) -> Iterator[str]:
        """
        Async iterator over tokens as they become available.
        
        This is a non-blocking async iterator that yields tokens until the stream is finished.
        """
        while True:
            token = await self.async_queue.get()
            if token is None:  # Stream finished
                break
            yield token
    
    def get_generated_text(self) -> str:
        """Get all generated text as a single string."""
        return "".join(self.all_tokens)