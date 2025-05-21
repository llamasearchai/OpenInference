"""
KV cache management for optimizing LLM inference memory usage.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class KeyValueCacheManager:
    """
    Manages key-value caches for transformer language models.
    
    Provides utilities for cache pruning, sharing, and optimization
    to reduce memory usage and improve throughput.
    """
    
    def __init__(self,
                max_cache_size_mb: int = 1024,
                prune_threshold: float = 0.8,
                device: str = "cuda"):
        """
        Initialize KV cache manager.
        
        Args:
            max_cache_size_mb: Maximum cache size in MB
            prune_threshold: Memory threshold to trigger pruning
            device: Device to store cache tensors
        """
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.prune_threshold = prune_threshold
        self.device = device
        
        # Cache storage
        self.cache_storage = {}
        self.request_to_cache = {}
        self.cache_sizes = {}
        self.last_access_time = {}
        self.current_size_bytes = 0
        
        # For tracking sequence reuse
        self.prefix_tree = {}
        
        # Stats
        self.stats = {
            "total_cache_hits": 0,
            "total_cache_misses": 0,
            "total_cache_evictions": 0,
            "total_cache_prunes": 0,
            "prefix_sharing_hits": 0,
            "memory_saved_mb": 0
        }
    
    def _get_tensor_size_bytes(self, tensor: torch.Tensor) -> int:
        """Calculate memory size of a tensor in bytes."""
        return tensor.element_size() * tensor.nelement()
    
    def _get_kv_cache_size_bytes(self, kv_cache: Any) -> int:
        """Calculate total size of a KV cache in bytes."""
        total_bytes = 0
        
        # Handle different cache formats
        if isinstance(kv_cache, tuple) and all(isinstance(layer, tuple) for layer in kv_cache):
            # Format: Tuple of layers, each layer is a tuple of (key, value) tensors
            for layer in kv_cache:
                for tensor in layer:
                    total_bytes += self._get_tensor_size_bytes(tensor)
        elif isinstance(kv_cache, dict) and "key_states" in kv_cache and "value_states" in kv_cache:
            # Format: Dict with key_states and value_states
            total_bytes += self._get_tensor_size_bytes(kv_cache["key_states"])
            total_bytes += self._get_tensor_size_bytes(kv_cache["value_states"])
        else:
            logger.warning(f"Unknown KV cache format: {type(kv_cache)}")
        
        return total_bytes
    
    def add_kv_cache(self, 
                    request_id: str, 
                    sequence_id: str, 
                    kv_cache: Any, 
                    token_count: int) -> bool:
        """
        Add a KV cache to the cache store.
        
        Args:
            request_id: ID of the request
            sequence_id: ID of the sequence (used for prefix sharing)
            kv_cache: Key-value cache to store
            token_count: Number of tokens in the sequence
            
        Returns:
            success: Whether cache was successfully added
        """
        try:
            # Calculate cache size
            cache_size_bytes = self._get_kv_cache_size_bytes(kv_cache)
            
            # Check if we need to prune first
            if self.current_size_bytes + cache_size_bytes > self.max_cache_size_bytes * self.prune_threshold:
                self._prune_caches()
            
            # Check if we still have space
            if self.current_size_bytes + cache_size_bytes > self.max_cache_size_bytes:
                logger.warning(f"Cache for {request_id} exceeds available space even after pruning")
                self.stats["total_cache_misses"] += 1
                return False
            
            # Store cache
            cache_id = f"{request_id}_{int(time.time() * 1000)}"
            self.cache_storage[cache_id] = kv_cache
            self.request_to_cache[request_id] = cache_id
            self.cache_sizes[cache_id] = cache_size_bytes
            self.last_access_time[cache_id] = time.time()
            
            # Update total size
            self.current_size_bytes += cache_size_bytes
            
            # Update prefix sharing tree
            if sequence_id:
                self._update_prefix_tree(sequence_id, cache_id, token_count)
            
            logger.debug(f"Added KV cache for {request_id} (size: {cache_size_bytes / 1024 / 1024:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding KV cache for {request_id}: {str(e)}")
            return False
    
    def get_kv_cache(self, request_id: str) -> Optional[Any]:
        """
        Retrieve a KV cache by request ID.
        
        Args:
            request_id: ID of the request
            
        Returns:
            kv_cache: The stored KV cache or None if not found
        """
        try:
            cache_id = self.request_to_cache.get(request_id)
            if not cache_id or cache_id not in self.cache_storage:
                self.stats["total_cache_misses"] += 1
                return None
            
            # Update last access time
            self.last_access_time[cache_id] = time.time()
            
            self.stats["total_cache_hits"] += 1
            return self.cache_storage[cache_id]
            
        except Exception as e:
            logger.error(f"Error retrieving KV cache for {request_id}: {str(e)}")
            return None
    
    def find_shared_prefix_cache(self, 
                               prompt_hash: str, 
                               token_count: int) -> Tuple[Optional[Any], int]:
        """
        Find a cached KV state that shares a prefix with the current prompt.
        
        Args:
            prompt_hash: Hash of the prompt tokens
            token_count: Number of tokens in the prompt
            
        Returns:
            cache_tuple: (kv_cache, prefix_length) or (None, 0) if no match
        """
        try:
            # Check if we have this prefix in the tree
            node = self.prefix_tree
            for i in range(min(len(prompt_hash), 8)):  # Use first 8 chars of hash for tree traversal
                if prompt_hash[i] not in node:
                    return None, 0
                node = node[prompt_hash[i]]
            
            # Find best matching cache
            best_match = None
            best_prefix_len = 0
            
            for cache_info in node.get("caches", []):
                cache_id = cache_info["cache_id"]
                prefix_len = cache_info["token_count"]
                
                # Skip if cache no longer exists
                if cache_id not in self.cache_storage:
                    continue
                
                # Check if this is a better match
                if prefix_len <= token_count and prefix_len > best_prefix_len:
                    best_match = cache_id
                    best_prefix_len = prefix_len
            
            if best_match:
                # Update last access time
                self.last_access_time[best_match] = time.time()
                self.stats["prefix_sharing_hits"] += 1
                
                # Calculate memory saved
                self.stats["memory_saved_mb"] += (
                    self.cache_sizes[best_match] * (best_prefix_len / token_count) / 1024 / 1024
                )
                
                return self.cache_storage[best_match], best_prefix_len
            
            return None, 0
            
        except Exception as e:
            logger.error(f"Error finding shared prefix: {str(e)}")
            return None, 0
    
    def _update_prefix_tree(self, sequence_id: str, cache_id: str, token_count: int):
        """
        Update the prefix sharing tree with a new cache entry.
        
        Args:
            sequence_id: ID/hash of the sequence
            cache_id: ID of the cache
            token_count: Number of tokens in the sequence
        """
        # Navigate/create path in tree
        node = self.prefix_tree
        for i in range(min(len(sequence_id), 8)):  # Use first 8 chars for tree
            if sequence_id[i] not in node:
                node[sequence_id[i]] = {}
            node = node[sequence_id[i]]
        
        # Add caches list if not exists
        if "caches" not in node:
            node["caches"] = []
        
        # Add this cache
        node["caches"].append({
            "cache_id": cache_id,
            "token_count": token_count
        })
        
        # Limit number of caches per node
        if len(node["caches"]) > 5:  # Keep only 5 most recent caches
            node["caches"] = node["caches"][-5:]
    
    def remove_kv_cache(self, request_id: str) -> bool:
        """
        Remove a KV cache from the store.
        
        Args:
            request_id: ID of the request
            
        Returns:
            success: Whether cache was successfully removed
        """
        try:
            cache_id = self.request_to_cache.get(request_id)
            if not cache_id:
                return False
            
            # Free memory
            if cache_id in self.cache_storage:
                cache_size = self.cache_sizes.get(cache_id, 0)
                self.current_size_bytes -= cache_size
                
                # Remove from storage
                del self.cache_storage[cache_id]
                del self.cache_sizes[cache_id]
                del self.last_access_time[cache_id]
            
            # Remove from request mapping
            del self.request_to_cache[request_id]
            
            # Note: We do not remove from prefix tree as it might be complex
            # The tree entries will be cleaned up when caches no longer exist
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing KV cache for {request_id}: {str(e)}")
            return False
    
    def _prune_caches(self):
        """
        Prune least recently used caches to free up memory.
        """
        try:
            # Skip if no caches to prune
            if not self.cache_storage:
                return
            
            # Sort caches by last access time
            sorted_caches = sorted(
                self.last_access_time.items(),
                key=lambda x: x[1]  # Sort by timestamp
            )
            
            # Calculate target size (free up to 40% of max size)
            target_size = int(self.max_cache_size_bytes * 0.6)
            
            # Remove caches until we're under target
            freed_bytes = 0
            pruned_count = 0
            
            for cache_id, _ in sorted_caches:
                if self.current_size_bytes <= target_size:
                    break
                
                # Get cache size
                cache_size = self.cache_sizes.get(cache_id, 0)
                
                # Remove from all tracking structures
                if cache_id in self.cache_storage:
                    del self.cache_storage[cache_id]
                    del self.cache_sizes[cache_id]
                    del self.last_access_time[cache_id]
                    
                    # Update counters
                    self.current_size_bytes -= cache_size
                    freed_bytes += cache_size
                    pruned_count += 1
                    
                    # Find and remove from request mapping
                    for req_id, c_id in list(self.request_to_cache.items()):
                        if c_id == cache_id:
                            del self.request_to_cache[req_id]
            
            # Update stats
            self.stats["total_cache_evictions"] += pruned_count
            self.stats["total_cache_prunes"] += 1
            
            logger.info(f"Pruned {pruned_count} caches, freed {freed_bytes / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error pruning caches: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about KV cache management."""
        stats = dict(self.stats)
        stats.update({
            "current_cache_count": len(self.cache_storage),
            "current_cache_size_mb": self.current_size_bytes / 1024 / 1024,
            "max_cache_size_mb": self.max_cache_size_bytes / 1024 / 1024,
            "cache_utilization": self.current_size_bytes / max(1, self.max_cache_size_bytes)
        })
        
        if stats["total_cache_hits"] + stats["total_cache_misses"] > 0:
            stats["cache_hit_ratio"] = (
                stats["total_cache_hits"] / 
                (stats["total_cache_hits"] + stats["total_cache_misses"])
            )
        else:
            stats["cache_hit_ratio"] = 0
            
        return stats