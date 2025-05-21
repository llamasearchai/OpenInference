"""
Flash Attention implementation for efficient memory usage and computation.
"""

import logging
import math
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class FlashAttention(nn.Module):
    """
    Flash Attention implementation for efficient attention computation.
    
    Implements the Flash Attention algorithm that reduces memory usage
    and improves performance by avoiding storing the full attention matrix.
    """
    
    def __init__(self,
                head_dim: int,
                num_heads: int,
                dropout_prob: float = 0.0,
                causal: bool = True,
                block_size: int = 64,
                device: str = "cuda"):
        """
        Initialize Flash Attention.
        
        Args:
            head_dim: Dimension of each attention head
            num_heads: Number of attention heads
            dropout_prob: Attention dropout probability
            causal: Whether to use causal masking
            block_size: Block size for tiled computation
            device: Device to run computation on
        """
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.causal = causal
        self.block_size = block_size
        self.device = device
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # Check if we can use the optimized version
        self._use_optimized = self._check_optimized_version()
    
    def _check_optimized_version(self) -> bool:
        """Check if optimized FlashAttention implementation is available."""
        try:
            # Check for FlashAttention CUDA implementation
            import flash_attn
            from flash_attn.flash_attention import FlashAttention as FlashAttnImpl
            logger.info("Using optimized FlashAttention implementation")
            self.flash_attn = FlashAttnImpl
            return True
        except ImportError:
            logger.info("Optimized FlashAttention not available, using PyTorch implementation")
            return False
    
    def forward(self,
               query: torch.Tensor,
               key: torch.Tensor,
               value: torch.Tensor,
               attn_mask: Optional[torch.Tensor] = None,
               key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention using FlashAttention algorithm.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, num_heads, head_dim]
            key: Key tensor [batch_size, seq_len_k, num_heads, head_dim]
            value: Value tensor [batch_size, seq_len_k, num_heads, head_dim]
            attn_mask: Optional attention mask [batch_size, num_heads, seq_len_q, seq_len_k]
            key_padding_mask: Optional key padding mask [batch_size, seq_len_k]
            
        Returns:
            output: Attention output [batch_size, seq_len_q, num_heads, head_dim]
        """
        # Use optimized implementation if available
        if self._use_optimized:
            return self._forward_optimized(query, key, value, attn_mask, key_padding_mask)
        else:
            return self._forward_pytorch(query, key, value, attn_mask, key_padding_mask)
    
    def _forward_optimized(self,
                        query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        attn_mask: Optional[torch.Tensor] = None,
                        key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using optimized FlashAttention implementation."""
        # Reshape inputs
        batch_size, seq_len_q, num_heads, head_dim = query.shape
        _, seq_len_k, _, _ = key.shape
        
        # Prepare inputs for FlashAttention
        # FlashAttention expects shape [batch_size, seq_len, num_heads, head_dim]
        # which is already what we have
        
        # Convert padding mask to attention mask if needed
        if key_padding_mask is not None:
            # Convert padding mask [batch_size, seq_len_k] to attention mask
            # [batch_size, 1, 1, seq_len_k]
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            padding_mask = padding_mask.expand(-1, -1, seq_len_q, -1)
            
            # Combine with existing attention mask or create new one
            if attn_mask is not None:
                attn_mask = attn_mask.logical_or(padding_mask)
            else:
                attn_mask = padding_mask
        
        # Apply Flash Attention
        # Note: flash_attn expects attention mask to be 0 for tokens to attend to
        # and 1 for tokens to ignore, which is the opposite of PyTorch convention
        dropout_p = self.dropout_prob if self.training else 0.0
        
        # Flip the mask if it exists
        if attn_mask is not None:
            # Convert boolean mask to float
            attn_mask = ~attn_mask if attn_mask.dtype == torch.bool else 1.0 - attn_mask
        
        # Call the optimized implementation
        output = self.flash_attn(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            causal=self.causal
        )
        
        return output
    
    def _forward_pytorch(self,
                      query: torch.Tensor,
                      key: torch.Tensor,
                      value: torch.Tensor,
                      attn_mask: Optional[torch.Tensor] = None,
                      key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass using PyTorch block-based implementation.
        
        This implementation processes attention in blocks to reduce memory usage, similar to the original FlashAttention
        algorithm, but implemented in native PyTorch.
        """
        # Extract shapes
        batch_size, seq_len_q, num_heads, head_dim = query.shape
        _, seq_len_k, _, _ = key.shape
        
        # Prepare output tensor
        output = torch.zeros_like(query)
        
        # Prepare normalization tensor to accumulate attention weights
        normalizer = torch.zeros((batch_size, seq_len_q, num_heads, 1), device=query.device)
        
        # Prepare causal mask if needed
        causal_mask = None
        if self.causal:
            # Create a lower triangular mask for causal attention
            causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=query.device)).bool()
            causal_mask = causal_mask.view(1, 1, seq_len_q, seq_len_k)
        
        # Combine masks if needed
        mask = None
        if causal_mask is not None or attn_mask is not None or key_padding_mask is not None:
            mask = torch.ones((batch_size, num_heads, seq_len_q, seq_len_k), dtype=torch.bool, device=query.device)
            
            if causal_mask is not None:
                mask = mask & causal_mask
            
            if attn_mask is not None:
                mask = mask & attn_mask
            
            if key_padding_mask is not None:
                # Convert padding mask [batch_size, seq_len_k] to attention mask
                # [batch_size, 1, 1, seq_len_k]
                padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                padding_mask = padding_mask.expand(-1, num_heads, seq_len_q, -1)
                mask = mask & ~padding_mask
        
        # Track maximum attention score for numerical stability
        max_score = torch.ones((batch_size, seq_len_q, num_heads, 1), device=query.device) * -1e10
        
        # Process in blocks to reduce memory usage
        for i in range(0, seq_len_k, self.block_size):
            # Get the current key block
            k_block_end = min(i + self.block_size, seq_len_k)
            key_block = key[:, i:k_block_end]
            value_block = value[:, i:k_block_end]
            
            # Process query blocks
            for j in range(0, seq_len_q, self.block_size):
                # Get the current query block
                q_block_end = min(j + self.block_size, seq_len_q)
                query_block = query[:, j:q_block_end]
                
                # Compute attention scores for this block
                # [batch_size, q_block_size, num_heads, k_block_size]
                scores = torch.matmul(query_block, key_block.transpose(-1, -2))
                scores = scores * self.scale
                
                # Apply mask if needed
                if mask is not None:
                    mask_block = mask[:, :, j:q_block_end, i:k_block_end]
                    scores = scores.masked_fill(~mask_block, -1e10)
                
                # Update max score for stability
                block_max_score = torch.max(scores, dim=-1, keepdim=True)[0]
                new_max_score = torch.maximum(max_score[:, j:q_block_end], block_max_score)
                
                # Scale current output and normalizer by exp(max_old - max_new)
                # for numerical stability
                exp_diff = torch.exp(max_score[:, j:q_block_end] - new_max_score)
                output[:, j:q_block_end] = output[:, j:q_block_end] * exp_diff
                normalizer[:, j:q_block_end] = normalizer[:, j:q_block_end] * exp_diff
                
                # Update max score
                max_score[:, j:q_block_end] = new_max_score
                
                # Apply softmax and dropout
                exp_scores = torch.exp(scores - new_max_score)
                
                if self.training and self.dropout_prob > 0:
                    dropout_mask = torch.bernoulli(
                        torch.full_like(exp_scores, 1 - self.dropout_prob)
                    ).to(exp_scores)
                    exp_scores = exp_scores * dropout_mask / (1 - self.dropout_prob)
                
                # Update output and normalizer
                # matmul: [batch_size, q_block_size, num_heads, k_block_size] x [batch_size, k_block_size, num_heads, head_dim]
                # -> [batch_size, q_block_size, num_heads, head_dim]
                output[:, j:q_block_end] = output[:, j:q_block_end] + torch.matmul(
                    exp_scores, value_block
                )
                normalizer[:, j:q_block_end] = normalizer[:, j:q_block_end] + exp_scores.sum(dim=-1, keepdim=True)
        
        # Normalize the output
        output = output / (normalizer + 1e-8)
        
        return output

class FlashMHA(nn.Module):
    """
    Multi-head attention using FlashAttention.
    
    A drop-in replacement for standard multi-head attention that uses
    the FlashAttention algorithm for improved efficiency.
    """
    
    def __init__(self,
                hidden_dim: int,
                num_heads: int,
                dropout_prob: float = 0.0,
                causal: bool = True,
                bias: bool = True,
                device: str = "cuda"):
        """
        Initialize FlashMHA.
        
        Args:
            hidden_dim: Model hidden dimension
            num_heads: Number of attention heads
            dropout_prob: Attention dropout probability
            causal: Whether to use causal masking
            bias: Whether to use bias in linear projections
            device: Device to run computation on
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout_prob = dropout_prob
        self.causal = causal
        
        # Check if dimensions work out
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias, device=device)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias, device=device)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias, device=device)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias, device=device)
        
        # Flash attention implementation
        self.flash_attn = FlashAttention(
            head_dim=self.head_dim,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            causal=causal,
            device=device
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters."""
        # Use Xavier initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self,
               query: torch.Tensor,
               key: Optional[torch.Tensor] = None,
               value: Optional[torch.Tensor] = None,
               attn_mask: Optional[torch.Tensor] = None,
               key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-head attention using FlashAttention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, hidden_dim]
            key: Optional key tensor, defaults to query
            value: Optional value tensor, defaults to key
            attn_mask: Optional attention mask [batch_size, num_heads, seq_len_q, seq_len_k]
            key_padding_mask: Optional key padding mask [batch_size, seq_len_k]
            
        Returns:
            output: Attention output [batch_size, seq_len_q, hidden_dim]
        """
        # Default key and value to query if not provided
        key = query if key is None else key
        value = key if value is None else value
        
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        # Apply flash attention
        attn_output = self.flash_attn(q, k, v, attn_mask, key_padding_mask)
        
        # Reshape and project back
        attn_output = attn_output.reshape(batch_size, seq_len_q, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        return output


def replace_attention_with_flash_attention(model: nn.Module, causal: bool = True) -> nn.Module:
    """
    Replace standard attention modules with FlashAttention.
    
    Args:
        model: PyTorch model
        causal: Whether to use causal masking
        
    Returns:
        model: Updated model with FlashAttention
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.MultiheadAttention):
            logger.info(f"Replacing MultiheadAttention module {name} with FlashMHA")
            
            # Create FlashMHA with same parameters
            flash_mha = FlashMHA(
                hidden_dim=module.embed_dim,
                num_heads=module.num_heads,
                dropout_prob=module.dropout,
                causal=causal,
                bias=module.in_proj_bias is not None,
                device=next(module.parameters()).device
            )
            
            # Replace module
            setattr(model, name, flash_mha)
        else:
            # Recursively process child modules
            replace_attention_with_flash_attention(module, causal)
    
    return model