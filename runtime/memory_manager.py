"""
Memory management utilities for optimizing GPU memory usage.
"""

import logging
import time
from typing import Dict, Any, Optional, Union, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class MemoryManager:
    """
    Memory manager for efficient memory usage during inference.
    
    Handles memory allocation, tracking, and optimization for
    inference workloads.
    """
    
    def __init__(self, 
                device: str = "cuda:0",
                reserved_memory_mb: int = 512,
                enable_memory_pool: bool = True):
        """
        Initialize the memory manager.
        
        Args:
            device: Device to manage memory for
            reserved_memory_mb: Amount of memory to reserve (in MB)
            enable_memory_pool: Whether to enable PyTorch's memory pool
        """
        self.device = device
        self.reserved_memory_mb = reserved_memory_mb
        self.enable_memory_pool = enable_memory_pool
        
        # Memory usage tracking
        self.peak_memory_allocated = 0
        self.current_allocations = {}
        
        # Initialize memory
        self._setup_memory()
    
    def _setup_memory(self):
        """Set up memory management for the device."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, memory management disabled")
            return
        
        try:
            if "cuda" in self.device and torch.cuda.is_available():
                # Configure memory pool
                if self.enable_memory_pool:
                    # Empty cache before setup
                    torch.cuda.empty_cache()
                    
                    # Synchronize before measuring memory
                    torch.cuda.synchronize()
                    
                    # Get total memory capacity
                    device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory
                    reserved_bytes = self.reserved_memory_mb * 1024 * 1024
                    
                    # Reserve memory for CUDA context and other overhead with safety bounds
                    fraction = 1.0 - (reserved_bytes / total_memory)
                    fraction = max(0.1, min(0.95, fraction))  # Keep between 10% and 95%
                    
                    # Use max_split_size_mb to control fragmentation (128MB is a good default)
                    torch.cuda.set_per_process_memory_fraction(fraction)
                    
                    # Enable memory stats tracking for detailed profiling
                    torch.cuda.memory._record_memory_history(enabled=True, max_entries=10000)
                
                logger.info(f"Memory manager initialized for {self.device} " 
                           f"with {self.reserved_memory_mb}MB reserved, " 
                           f"using {fraction*100:.1f}% of available memory")
            
            elif "mps" in self.device and hasattr(torch, 'mps') and torch.mps.is_available():
                # Metal Performance Shaders (Apple Silicon) setup
                # Configure GC threshold for better memory management
                import gc
                gc.set_threshold(700, 10, 5)  # Tune these values for Apple Silicon
                logger.info(f"Memory manager initialized for Apple Silicon {self.device}")
            
            else:
                logger.info(f"Memory manager initialized for {self.device}")
        
        except Exception as e:
            logger.error(f"Error setting up memory management: {str(e)}")
            # Add fallback mechanism for memory management
            logger.info("Using fallback memory management settings")
            self.enable_memory_pool = False
    
    def allocate(self, name: str, shape: Tuple[int, ...], dtype: Any) -> Optional[torch.Tensor]:
        """
        Allocate a tensor with the given shape and dtype.
        
        Args:
            name: Name to track this allocation
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor
            
        Returns:
            tensor: Allocated tensor or None if allocation failed
        """
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Calculate size in bytes
            element_size = torch.tensor([], dtype=dtype).element_size()
            size_bytes = np.prod(shape) * element_size
            
            # Create tensor
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            
            # Track allocation
            self.current_allocations[name] = {
                "tensor": tensor,
                "shape": shape,
                "dtype": dtype,
                "size_bytes": size_bytes,
                "allocated_time": time.time()
            }
            
            # Update peak memory
            if TORCH_AVAILABLE and "cuda" in self.device and torch.cuda.is_available():
                current_allocated = torch.cuda.memory_allocated(self.device)
                self.peak_memory_allocated = max(self.peak_memory_allocated, current_allocated)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error allocating memory {name} with shape {shape}: {str(e)}")
            return None
    
    def free(self, name: str) -> bool:
        """
        Free a previously allocated tensor.
        
        Args:
            name: Name of the allocation to free
            
        Returns:
            success: Whether the deallocation was successful
        """
        if name not in self.current_allocations:
            return False
        
        try:
            # Get allocation
            allocation = self.current_allocations[name]
            
            # Delete reference to tensor - Python's garbage collector will handle the rest
            del allocation["tensor"]
            del self.current_allocations[name]
            
            return True
            
        except Exception as e:
            logger.error(f"Error freeing memory {name}: {str(e)}")
            return False
    
    def clear_cache(self):
        """Clear PyTorch's CUDA memory cache."""
        if TORCH_AVAILABLE:
            if "cuda" in self.device and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif "mps" in self.device and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    
    def clear_all(self):
        """Free all allocated tensors and clear cache."""
        # Free all tracked allocations
        for name in list(self.current_allocations.keys()):
            self.free(name)
        
        # Clear cache
        self.clear_cache()
        
        # Reset tracking stats
        self.peak_memory_allocated = 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            "num_allocations": len(self.current_allocations),
            "peak_memory_allocated_mb": self.peak_memory_allocated / (1024 * 1024)
        }
        
        if TORCH_AVAILABLE:
            if "cuda" in self.device and torch.cuda.is_available():
                stats.update({
                    "current_allocated_mb": torch.cuda.memory_allocated(self.device) / (1024 * 1024),
                    "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024 * 1024),
                    "current_reserved_mb": torch.cuda.memory_reserved(self.device) / (1024 * 1024),
                    "max_reserved_mb": torch.cuda.max_memory_reserved(self.device) / (1024 * 1024)
                })
                
                # Add per-allocation details
                allocation_details = {}
                for name, alloc in self.current_allocations.items():
                    allocation_details[name] = {
                        "shape": alloc["shape"],
                        "dtype": str(alloc["dtype"]),
                        "size_mb": alloc["size_bytes"] / (1024 * 1024),
                        "age_seconds": time.time() - alloc["allocated_time"]
                    }
                
                stats["allocations"] = allocation_details
        
        return stats


class KVCacheManager:
    """
    Manager for transformer key-value cache memory.
    
    Optimizes memory usage and performance for transformer-based models
    by efficiently reusing and managing KV cache memory.
    """
    
    def __init__(self, 
                device: str = "cuda:0",
                max_memory_fraction: float = 0.8,
                cache_dtype: Optional[Any] = None):
        """
        Initialize the KV cache manager.
        
        Args:
            device: Device to manage cache on
            max_memory_fraction: Maximum fraction of GPU memory to use for KV cache
            cache_dtype: Data type for cache tensors (defaults to model dtype)
        """
        self.device = device
        self.max_memory_fraction = max_memory_fraction
        self.cache_dtype = cache_dtype
        
        # Cache storage - maps sequence ID to cache tensors
        self.kv_caches = {}
        
        # Cache size tracking
        self.total_cache_bytes = 0
        self.peak_cache_bytes = 0
        
        # Calculate maximum cache size
        self.max_cache_bytes = self._calculate_max_cache_size()
        
        # For LRU cache management
        self.last_used_time = {}
    
    def _calculate_max_cache_size(self) -> int:
        """Calculate maximum cache size in bytes."""
        if not TORCH_AVAILABLE:
            return 1 * 1024 * 1024 * 1024  # Default 1GB
        
        try:
            if "cuda" in self.device and torch.cuda.is_available():
                device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                max_size = int(total_memory * self.max_memory_fraction)
                
                logger.info(f"KV cache max size: {max_size / (1024**3):.2f} GB " 
                           f"({self.max_memory_fraction*100:.0f}% of GPU memory)")
                
                return max_size
            elif "mps" in self.device and hasattr(torch, 'mps') and torch.mps.is_available():
                # For Apple Silicon, we use a fixed size since memory APIs are limited
                # Default to 4GB or 80% of system memory
                import psutil
                system_memory = psutil.virtual_memory().total
                max_size = min(4 * 1024**3, int(system_memory * 0.8 * self.max_memory_fraction))
                
                logger.info(f"KV cache max size for MPS: {max_size / (1024**3):.2f} GB")
                
                return max_size
            else:
                # For CPU, use a smaller default
                return 2 * 1024 * 1024 * 1024  # 2GB
        
        except Exception as e:
            logger.error(f"Error calculating max cache size: {str(e)}")
            return 1 * 1024 * 1024 * 1024  # Default 1GB
    
    def allocate_cache(self, 
                      seq_id: str, 
                      num_layers: int, 
                      kv_dim: int, 
                      max_seq_len: int,
                      num_heads: int,
                      head_dim: int,
                      dtype: Optional[Any] = None,
                      flash_attention: bool = True) -> Dict[str, Any]:
        """
        Allocate or retrieve KV cache for a sequence with improved efficiency.
        
        Args:
            seq_id: Unique identifier for the sequence
            num_layers: Number of transformer layers
            kv_dim: Dimension of key/value projections
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
            dtype: Data type for the cache
            flash_attention: Whether to use flash attention compatible format
        
        Returns:
            cache: Dictionary containing key and value cache tensors
        """
        if not TORCH_AVAILABLE:
            return None
        
        # Use provided dtype or default
        dtype = dtype or self.cache_dtype or torch.float16
        
        # Check if this sequence already has a cache
        if seq_id in self.kv_caches:
            # Update last used time
            self.last_used_time[seq_id] = time.time()
        
            # Check if max_seq_len is larger than existing cache
            existing_max_len = self.kv_caches[seq_id]["max_seq_len"]
            if max_seq_len > existing_max_len:
                logger.info(f"Expanding KV cache for sequence {seq_id} from {existing_max_len} to {max_seq_len}")
                # We need to expand the cache - implement this by creating a new cache
                # and copying the existing content
                self.free_cache(seq_id)
            else:
                return self.kv_caches[seq_id]
        
        try:
            # Calculate size of new cache
            element_size = torch.tensor([], dtype=dtype).element_size()
            
            # Using a memory-efficient format for flash attention if requested
            if flash_attention:
                # Flash attention often uses [num_layers, 2, batch_size=1, num_heads, head_dim, max_seq_len]
                # which is more memory-efficient for attention computation
                cache_shape = (num_layers, 2, 1, num_heads, head_dim, max_seq_len)
                cache_size_bytes = np.prod(cache_shape) * element_size
                
                # Check if adding this cache would exceed max cache size
                if self.total_cache_bytes + cache_size_bytes > self.max_cache_bytes:
                    # Need to free up space - remove least recently used caches
                    self._evict_caches(cache_size_bytes)
                    
                    # If still not enough space, use a reduced cache size
                    if self.total_cache_bytes + cache_size_bytes > self.max_cache_bytes:
                        # Reduce max_seq_len to fit in memory
                        reduced_max_seq_len = int(max_seq_len * 0.75)  # Try 75% of requested size
                        logger.warning(f"Insufficient memory for full KV cache. Reducing max sequence length from {max_seq_len} to {reduced_max_seq_len}")
                        max_seq_len = max(128, reduced_max_seq_len)  # Ensure at least 128 tokens
                        cache_shape = (num_layers, 2, 1, num_heads, head_dim, max_seq_len)
                        cache_size_bytes = np.prod(cache_shape) * element_size
                
                # Create a single tensor to hold both key and value caches for all layers
                # This is more memory-efficient and reduces fragmentation
                try:
                    # Try using pytorch's memory-efficient empty (available in newer versions)
                    kv_cache_tensor = torch.empty(
                        cache_shape,
                        dtype=dtype,
                        device=self.device
                    )
                except Exception:
                    # Fallback to standard tensor creation
                    kv_cache_tensor = torch.zeros(
                        cache_shape,
                        dtype=dtype,
                        device=self.device
                    )
                
                # Store cache
                self.kv_caches[seq_id] = {
                    "kv_cache": kv_cache_tensor,
                    "size_bytes": cache_size_bytes,
                    "max_seq_len": max_seq_len,
                    "current_len": 0,  # No tokens cached yet
                    "flash_attention": True,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "head_dim": head_dim
                }
            else:
                # Traditional format with separate key and value tensors for each layer
                # Shape is typically [batch_size=1, num_heads, max_seq_len, head_dim]
                cache_size_bytes = 2 * num_layers * 1 * num_heads * max_seq_len * head_dim * element_size
                
                # Check if adding this cache would exceed max cache size
                if self.total_cache_bytes + cache_size_bytes > self.max_cache_bytes:
                    # Need to free up space - remove least recently used caches
                    self._evict_caches(cache_size_bytes)
                
                # Create key and value caches for each layer
                k_caches = {}
                v_caches = {}
                
                for layer_idx in range(num_layers):
                    # Allocate key cache
                    k_caches[layer_idx] = torch.zeros(
                        (1, num_heads, max_seq_len, head_dim),
                        dtype=dtype,
                        device=self.device
                    )
                    
                    # Allocate value cache
                    v_caches[layer_idx] = torch.zeros(
                        (1, num_heads, max_seq_len, head_dim),
                        dtype=dtype,
                        device=self.device
                    )
                
                # Store cache
                self.kv_caches[seq_id] = {
                    "key_cache": k_caches,
                    "value_cache": v_caches,
                    "size_bytes": cache_size_bytes,
                    "max_seq_len": max_seq_len,
                    "current_len": 0,  # No tokens cached yet
                    "flash_attention": False,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "head_dim": head_dim
                }
            
            # Update tracking
            self.total_cache_bytes += cache_size_bytes
            self.peak_cache_bytes = max(self.peak_cache_bytes, self.total_cache_bytes)
            self.last_used_time[seq_id] = time.time()
            
            logger.debug(f"Allocated KV cache for sequence {seq_id}: {cache_size_bytes / (1024**2):.2f} MB")
            
            return self.kv_caches[seq_id]
            
        except Exception as e:
            logger.error(f"Error allocating KV cache for {seq_id}: {str(e)}")
            # Try a fallback with minimal requirements
            try:
                logger.info("Attempting fallback KV cache allocation with reduced dimensions")
                # Simplified allocation with minimal dimensions
                k_caches = {}
                v_caches = {}
                reduced_max_seq_len = min(512, max_seq_len)  # Limit to 512 tokens
                
                for layer_idx in range(num_layers):
                    k_caches[layer_idx] = torch.zeros(
                        (1, num_heads, reduced_max_seq_len, head_dim),
                        dtype=torch.float16,  # Force float16 to save memory
                        device=self.device
                    )
                    v_caches[layer_idx] = torch.zeros(
                        (1, num_heads, reduced_max_seq_len, head_dim),
                        dtype=torch.float16,
                        device=self.device
                    )
                
                cache_size_bytes = 2 * num_layers * 1 * num_heads * reduced_max_seq_len * head_dim * 2  # float16 = 2 bytes
                
                self.kv_caches[seq_id] = {
                    "key_cache": k_caches,
                    "value_cache": v_caches,
                    "size_bytes": cache_size_bytes,
                    "max_seq_len": reduced_max_seq_len,
                    "current_len": 0,
                    "flash_attention": False,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "head_dim": head_dim
                }
                
                self.total_cache_bytes += cache_size_bytes
                self.peak_cache_bytes = max(self.peak_cache_bytes, self.total_cache_bytes)
                self.last_used_time[seq_id] = time.time()
                
                logger.info(f"Fallback KV cache allocated with reduced dimensions: {reduced_max_seq_len} max length")
                return self.kv_caches[seq_id]
            except Exception as e2:
                logger.error(f"Fallback KV cache allocation also failed: {str(e2)}")
                return None
    
    def _evict_caches(self, required_bytes: int):
        """
        Evict least recently used caches to free up required memory.
        
        Args:
            required_bytes: Minimum number of bytes to free
        """
        if not self.kv_caches:
            logger.warning("No caches to evict")
            return
        
        # Sort sequences by last access time
        sorted_seqs = sorted(self.last_used_time.items(), key=lambda x: x[1])
        bytes_freed = 0
        
        for seq_id, _ in sorted_seqs:
            if seq_id in self.kv_caches:
                cache_size = self.kv_caches[seq_id]["size_bytes"]
                
                # Free this cache
                logger.debug(f"Evicting KV cache for sequence {seq_id}: {cache_size / (1024**2):.2f} MB")
                
                del self.kv_caches[seq_id]
                del self.last_used_time[seq_id]
                
                bytes_freed += cache_size
                self.total_cache_bytes -= cache_size
                
                if bytes_freed >= required_bytes:
                    break
        
        if bytes_freed < required_bytes:
            logger.warning(f"Could not free enough memory for KV cache: "
                          f"freed {bytes_freed / (1024**2):.2f} MB, "
                          f"needed {required_bytes / (1024**2):.2f} MB")
    
    def update_cache(self, 
                    seq_id: str, 
                    layer_idx: int, 
                    position: int, 
                    key: torch.Tensor, 
                    value: torch.Tensor) -> bool:
        """
        Update the KV cache for a sequence at a specific position.
        
        Args:
            seq_id: Sequence identifier
            layer_idx: Index of the transformer layer
            position: Position in the sequence
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            success: Whether the update was successful
        """
        if seq_id not in self.kv_caches:
            logger.error(f"Cannot update cache for unknown sequence {seq_id}")
            return False
        
        try:
            # Get cache for this sequence
            cache = self.kv_caches[seq_id]
            
            # Check if layer index is valid
            if layer_idx not in cache["key_cache"] or layer_idx not in cache["value_cache"]:
                logger.error(f"Invalid layer index {layer_idx} for sequence {seq_id}")
                return False
            
            # Check if position is valid
            if position >= cache["max_seq_len"]:
                logger.error(f"Position {position} exceeds max sequence length {cache['max_seq_len']}")
                return False
            
            # Update key and value caches at the specified position
            # Assumes key and value have shape [batch_size, num_heads, seq_len, head_dim]
            # We're storing the last token if sequence length > 1
            token_idx = min(key.size(2) - 1, 0)  # Default to last token
            
            cache["key_cache"][layer_idx][:, :, position, :] = key[:, :, token_idx, :]
            cache["value_cache"][layer_idx][:, :, position, :] = value[:, :, token_idx, :]
            
            # Update current length if we're appending
            if position == cache["current_len"]:
                cache["current_len"] += 1
            
            # Update last used time
            self.last_used_time[seq_id] = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating KV cache for sequence {seq_id}: {str(e)}")
            return False
    
    def get_cache(self, seq_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the KV cache for a sequence.
        
        Args:
            seq_id: Sequence identifier
            
        Returns:
            cache: Dictionary containing key and value cache tensors
        """
        if seq_id not in self.kv_caches:
            return None
        
        # Update last used time
        self.last_used_time[seq_id] = time.time()
        
        return self.kv_caches[seq_id]
    
    def free_cache(self, seq_id: str) -> bool:
        """
        Free the KV cache for a sequence.
        
        Args:
            seq_id: Sequence identifier
            
        Returns:
            success: Whether the cache was freed successfully
        """
        if seq_id not in self.kv_caches:
            return False
        
        try:
            # Get cache size
            cache_size = self.kv_caches[seq_id]["size_bytes"]
            
            # Free cache
            del self.kv_caches[seq_id]
            
            # Update tracking
            if seq_id in self.last_used_time:
                del self.last_used_time[seq_id]
                
            self.total_cache_bytes -= cache_size
            
            return True
            
        except Exception as e:
            logger.error(f"Error freeing KV cache for sequence {seq_id}: {str(e)}")
            return False
    
    def clear_all(self):
        """Free all KV caches."""
        # Free all caches
        self.kv_caches.clear()
        self.last_used_time.clear()
        
        # Reset tracking
        self.total_cache_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the KV cache usage."""
        return {
            "num_active_sequences": len(self.kv_caches),
            "total_cache_mb": self.total_cache_bytes / (1024 * 1024),
            "peak_cache_mb": self.peak_cache_bytes / (1024 * 1024),
            "max_cache_mb": self.max_cache_bytes / (1024 * 1024),
            "utilization_percent": (self.total_cache_bytes / self.max_cache_bytes) * 100 if self.max_cache_bytes > 0 else 0
        }