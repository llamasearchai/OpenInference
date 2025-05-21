"""
KV Cache Manager for efficient memory usage during LLM inference.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import heapq
import weakref

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class KVCacheEntry:
    """
    A single entry in the KV cache, representing one sequence.
    """
    
    def __init__(self,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                sequence_id: str,
                max_length: int,
                last_access_time: float,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize KV cache entry.
        
        Args:
            key_cache: Key cache tensor
            value_cache: Value cache tensor
            sequence_id: Unique sequence identifier
            max_length: Maximum sequence length
            last_access_time: Last access time (for LRU eviction)
            metadata: Optional metadata
        """
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.sequence_id = sequence_id
        self.max_length = max_length
        self.last_access_time = last_access_time
        self.current_len = 0
        self.metadata = metadata or {}
    
    def extend(self, key: torch.Tensor, value: torch.Tensor) -> bool:
        """
        Extend the cache with new key-value pairs.
        
        Args:
            key: New key tensor [num_layers, num_heads, seq_len, head_dim]
            value: New value tensor [num_layers, num_heads, seq_len, head_dim]
            
        Returns:
            success: Whether the extension was successful
        """
        new_tokens = key.size(2)
        new_len = self.current_len + new_tokens
        
        # Check if we have space
        if new_len > self.max_length:
            return False
        
        # Add new tokens to cache
        self.key_cache[:, :, self.current_len:new_len] = key
        self.value_cache[:, :, self.current_len:new_len] = value
        
        # Update length and access time
        self.current_len = new_len
        self.last_access_time = time.time()
        
        return True
    
    def get(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a slice of the KV cache.
        
        Args:
            start_idx: Optional start index
            end_idx: Optional end index
            
        Returns:
            key_slice: Slice of key cache
            value_slice: Slice of value cache
        """
        start = 0 if start_idx is None else start_idx
        end = self.current_len if end_idx is None else min(end_idx, self.current_len)
        
        # Update access time
        self.last_access_time = time.time()
        
        return self.key_cache[:, :, start:end], self.value_cache[:, :, start:end]
    
    def resize(self, new_max_length: int) -> bool:
        """
        Resize the cache to a new maximum length.
        
        Args:
            new_max_length: New maximum sequence length
            
        Returns:
            success: Whether resizing was successful
        """
        if new_max_length < self.current_len:
            return False
        
        # Create new cache tensors
        new_key_cache = torch.zeros(
            (self.key_cache.size(0), self.key_cache.size(1), new_max_length, self.key_cache.size(3)),
            dtype=self.key_cache.dtype,
            device=self.key_cache.device
        )
        new_value_cache = torch.zeros(
            (self.value_cache.size(0), self.value_cache.size(1), new_max_length, self.value_cache.size(3)),
            dtype=self.value_cache.dtype,
            device=self.value_cache.device
        )
        
        # Copy existing data
        new_key_cache[:, :, :self.current_len] = self.key_cache[:, :, :self.current_len]
        new_value_cache[:, :, :self.current_len] = self.value_cache[:, :, :self.current_len]
        
        # Update cache tensors
        self.key_cache = new_key_cache
        self.value_cache = new_value_cache
        self.max_length = new_max_length
        
        return True
    
    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        key_bytes = self.key_cache.element_size() * self.key_cache.nelement()
        value_bytes = self.value_cache.element_size() * self.value_cache.nelement()
        return key_bytes + value_bytes


class KVCacheManager:
    """
    Manages KV caches for multiple sequences during LLM inference.
    
    Implements efficient memory management strategies:
    - Pre-allocation of cache blocks
    - LRU-based eviction of inactive caches
    - Block reuse to reduce fragmentation
    - Automatic cache pruning for long sequences
    """
    
    def __init__(self,
                num_layers: int,
                num_heads: int,
                head_dim: int,
                max_total_tokens: int = 1_000_000,
                max_cache_entries: int = 256,
                dtype: torch.dtype = torch.float16,
                device: str = "cuda"):
        """
        Initialize KV cache manager.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            max_total_tokens: Maximum total tokens across all caches
            max_cache_entries: Maximum number of active cache entries
            dtype: Data type for cache tensors
            device: Device to store caches on
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_total_tokens = max_total_tokens
        self.max_cache_entries = max_cache_entries
        self.dtype = dtype
        self.device = device
        
        # Cache storage
        self.caches: Dict[str, KVCacheEntry] = {}
        
        # Memory tracking
        self.total_allocated_tokens = 0
        self.total_used_tokens = 0
        
        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "allocations": 0,
            "total_sequences": 0,
            "max_memory_used": 0
        }
        
        logger.info(f"Initialized KV cache manager with {max_total_tokens} max tokens")
    
    def allocate_cache(self, sequence_id: str, max_length: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Allocate a new KV cache for a sequence.
        
        Args:
            sequence_id: Unique sequence identifier
            max_length: Maximum sequence length
            metadata: Optional metadata
            
        Returns:
            success: Whether allocation was successful
        """
        # Check if cache already exists
        if sequence_id in self.caches:
            logger.debug(f"Cache for sequence {sequence_id} already exists")
            return True
        
        # Check if we've reached the maximum number of caches
        if len(self.caches) >= self.max_cache_entries:
            # Try to evict least recently used cache
            if not self._evict_lru_cache():
                logger.warning(f"Failed to allocate cache for sequence {sequence_id}: maximum entries reached")
                return False
        
        # Check if we've reached the maximum total tokens
        if self.total_allocated_tokens + max_length > self.max_total_tokens:
            # Try to evict caches to make room
            tokens_to_free = (self.total_allocated_tokens + max_length) - self.max_total_tokens
            if not self._free_cache_memory(tokens_to_free):
                logger.warning(f"Failed to allocate cache for sequence {sequence_id}: not enough memory")
                return False
        
        # Allocate new cache tensors
        try:
            key_cache = torch.zeros(
                (self.num_layers, self.num_heads, max_length, self.head_dim),
                dtype=self.dtype,
                device=self.device
            )
            
            value_cache = torch.zeros(
                (self.num_layers, self.num_heads, max_length, self.head_dim),
                dtype=self.dtype,
                device=self.device
            )
            
            # Create cache entry
            cache_entry = KVCacheEntry(
                key_cache=key_cache,
                value_cache=value_cache,
                sequence_id=sequence_id,
                max_length=max_length,
                last_access_time=time.time(),
                metadata=metadata
            )
            
            # Store cache
            self.caches[sequence_id] = cache_entry
            
            # Update tracking
            self.total_allocated_tokens += max_length
            self.metrics["allocations"] += 1
            self.metrics["total_sequences"] += 1
            self.metrics["max_memory_used"] = max(
                self.metrics["max_memory_used"],
                self.total_allocated_tokens
            )
            
            logger.debug(f"Allocated cache for sequence {sequence_id} with {max_length} tokens")
            return True
            
        except RuntimeError as e:
            logger.error(f"Failed to allocate cache for sequence {sequence_id}: {str(e)}")
            return False
    
    def extend_cache(self, sequence_id: str, key: torch.Tensor, value: torch.Tensor) -> bool:
        """
        Extend an existing cache with new key-value pairs.
        
        Args:
            sequence_id: Sequence identifier
            key: New key tensor [num_layers, num_heads, seq_len, head_dim]
            value: New value tensor [num_layers, num_heads, seq_len, head_dim]
            
        Returns:
            success: Whether extension was successful
        """
        # Check if cache exists
        if sequence_id not in self.caches:
            self.metrics["cache_misses"] += 1
            logger.warning(f"Cache for sequence {sequence_id} not found")
            return False
        
        # Get cache entry
        cache_entry = self.caches[sequence_id]
        
        # Try to extend
        if not cache_entry.extend(key, value):
            logger.warning(f"Failed to extend cache for sequence {sequence_id}: cache full")
            return False
        
        # Update tracking
        new_tokens = key.size(2)
        self.total_used_tokens += new_tokens
        self.metrics["cache_hits"] += 1
        
        return True
    
    def get_cache(self, sequence_id: str, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a slice of the KV cache for a sequence.
        
        Args:
            sequence_id: Sequence identifier
            start_idx: Optional start index
            end_idx: Optional end index
            
        Returns:
            cache: Tuple of (key_cache, value_cache) or None if not found
        """
        # Check if cache exists
        if sequence_id not in self.caches:
            self.metrics["cache_misses"] += 1
            return None
        
        # Get cache slice
        cache_entry = self.caches[sequence_id]
        key_slice, value_slice = cache_entry.get(start_idx, end_idx)
        
        self.metrics["cache_hits"] += 1
        return key_slice, value_slice
    
    def release_cache(self, sequence_id: str):
        """
        Release a cache when it's no longer needed.
        
        Args:
            sequence_id: Sequence identifier
        """
        if sequence_id in self.caches:
            cache_entry = self.caches[sequence_id]
            self.total_allocated_tokens -= cache_entry.max_length
            del self.caches[sequence_id]
            logger.debug(f"Released cache for sequence {sequence_id}")
    
    def _evict_lru_cache(self) -> bool:
        """
        Evict the least recently used cache.
        
        Returns:
            success: Whether eviction was successful
        """
        if not self.caches:
            return False
        
        # Find least recently used cache
        lru_id = min(self.caches.keys(), key=lambda k: self.caches[k].last_access_time)
        
        # Evict it
        self.release_cache(lru_id)
        self.metrics["evictions"] += 1
        
        return True
    
    def _free_cache_memory(self, tokens_to_free: int) -> bool:
        """
        Free cache memory by evicting caches.
        
        Args:
            tokens_to_free: Number of tokens to free
            
        Returns:
            success: Whether enough memory was freed
        """
        # Sort caches by last access time
        sorted_caches = sorted(
            self.caches.items(),
            key=lambda x: x[1].last_access_time
        )
        
        # Evict caches until we've freed enough memory
        freed_tokens = 0
        for seq_id, cache in sorted_caches:
            self.release_cache(seq_id)
            freed_tokens += cache.max_length
            self.metrics["evictions"] += 1
            
            if freed_tokens >= tokens_to_free:
                return True
        
        return freed_tokens >= tokens_to_free
    
    def prune_finished_sequences(self, active_sequences: List[str]):
        """
        Prune caches for sequences that are no longer active.
        
        Args:
            active_sequences: List of active sequence IDs
        """
        active_set = set(active_sequences)
        to_release = [seq_id for seq_id in self.caches if seq_id not in active_set]
        
        for seq_id in to_release:
            self.release_cache(seq_id)
    
    def clear_all(self):
        """Clear all caches."""
        self.caches.clear()
        self.total_allocated_tokens = 0
        self.total_used_tokens = 0
        logger.info("Cleared all KV caches")
    
    def memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        return sum(cache.memory_usage() for cache in self.caches.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = dict(self.metrics)
        metrics["num_active_caches"] = len(self.caches)
        metrics["allocated_tokens"] = self.total_allocated_tokens
        metrics["used_tokens"] = self.total_used_tokens
        metrics["memory_usage_bytes"] = self.memory_usage()
        
        # Calculate hit rate
        total_requests = metrics["cache_hits"] + metrics["cache_misses"]
        metrics["cache_hit_rate"] = metrics["cache_hits"] / total_requests if total_requests > 0 else 0.0
        
        return metrics