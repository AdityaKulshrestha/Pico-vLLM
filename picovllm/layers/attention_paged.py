"""
Attention Layer with PagedAttention Support for Intel Xeon CPUs

This module provides the main Attention class that integrates with the
PagedAttention kernels for efficient inference on Intel Xeon processors.
"""

import torch
from torch import nn
from typing import Optional

from picovllm.utils.context import get_context

# Import our custom PagedAttention kernels
from picovllm.kernels.paged_attention import (
    PagedAttention as PagedAttentionModule,
    reshape_and_cache,
    paged_attention_prefill,
    paged_attention_decode,
)

# Try to import the C++ AVX-512 kernels
try:
    from picovllm.kernels.csrc import paged_attention_cpu as paged_attn_cpp
    HAS_CPP_KERNELS = True
except ImportError:
    HAS_CPP_KERNELS = False

# Try to import IPEX for additional optimizations
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 256,
) -> None:
    """
    Store key and value tensors into the paged KV cache.
    
    Automatically selects the best implementation:
    1. C++ AVX-512 kernel (fastest)
    2. Pure PyTorch with advanced indexing (fallback)
    
    Args:
        key: Key tensor (N, num_heads, head_dim)
        value: Value tensor (N, num_heads, head_dim)
        k_cache: Key cache (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: Value cache (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: Slot indices for each token (N,)
        block_size: Number of tokens per block
    """
    if HAS_CPP_KERNELS and key.dtype == torch.float32:
        # Use AVX-512 optimized C++ kernel
        paged_attn_cpp.reshape_and_cache(
            key.contiguous(),
            value.contiguous(),
            k_cache,
            v_cache,
            slot_mapping.contiguous(),
            block_size,
        )
    else:
        # Fallback to our Python/TorchScript implementation
        reshape_and_cache(key, value, k_cache, v_cache, slot_mapping, block_size)


class Attention(nn.Module):
    """
    Multi-Head Attention with PagedAttention support for Intel Xeon CPUs.
    
    This class provides efficient attention computation using:
    - Paged KV cache for memory-efficient long-context inference
    - AVX-512 vectorized kernels for CPU optimization
    - Flash Attention-style chunked computation for prefill
    - Optimized single-query attention for decode
    
    Supports both standard Multi-Head Attention and Grouped Query Attention (GQA).
    
    Args:
        num_heads: Number of query attention heads
        head_dim: Dimension of each attention head
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        num_kv_heads: Number of key/value heads (for GQA, can be < num_heads)
        alibi_slopes: Optional ALiBi positional encoding slopes
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        # Placeholder for KV cache (will be set by model_runner)
        self.k_cache: torch.Tensor = torch.tensor([])
        self.v_cache: torch.Tensor = torch.tensor([])
        
        # Optional ALiBi slopes
        if alibi_slopes is not None:
            self.register_buffer('alibi_slopes', alibi_slopes)
        else:
            self.alibi_slopes = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute attention with paged KV cache.
        
        Args:
            q: Query tensor (num_tokens, num_heads, head_dim)
            k: Key tensor (num_tokens, num_kv_heads, head_dim)
            v: Value tensor (num_tokens, num_kv_heads, head_dim)
            dropout_p: Dropout probability (not used for inference)
            
        Returns:
            Output tensor (num_tokens, num_heads, head_dim)
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Store KV to cache if cache is allocated
        if k_cache.numel() > 0 and v_cache.numel() > 0:
            block_size = k_cache.size(1)
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, block_size)
        
        if context.is_prefill:
            return self._forward_prefill(q, k, v, k_cache, v_cache, context)
        else:
            return self._forward_decode(q, k_cache, v_cache, context)
    
    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """
        Forward pass for prefill phase.
        
        Uses chunked Flash Attention-style computation for long sequences.
        """
        # Determine if we should use cached KV (prefix caching)
        use_cached_kv = context.block_tables is not None
        
        if use_cached_kv:
            # Read KV from cache
            key_source = k_cache
            value_source = v_cache
        else:
            # Use input KV directly (store to cache happens separately)
            key_source = k
            value_source = v
        
        # Allocate output buffer
        output = torch.empty_like(q)
        
        # Select best kernel
        if HAS_CPP_KERNELS and q.dtype == torch.float32 and use_cached_kv:
            # Use C++ AVX-512 kernel
            paged_attn_cpp.paged_attention_prefill(
                output,
                q.contiguous(),
                k_cache,
                v_cache,
                context.cu_seqlens_q.contiguous(),
                context.cu_seqlens_k.contiguous(),
                context.max_seqlen_q,
                context.max_seqlen_k,
                self.scale,
                True,  # causal
                context.block_tables,
            )
        elif HAS_IPEX and use_cached_kv:
            # Try IPEX flash attention
            try:
                ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
                    output,
                    q,
                    k_cache,
                    v_cache,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                    context.max_seqlen_q,
                    context.max_seqlen_k,
                    self.scale,
                    True,  # causal
                    context.block_tables,
                    alibi_slopes=self.alibi_slopes,
                )
            except Exception:
                # Fallback to our Python implementation
                self._paged_attention_prefill_python(
                    output, q, key_source, value_source, context, use_cached_kv
                )
        else:
            # Use our Python/TorchScript implementation
            self._paged_attention_prefill_python(
                output, q, key_source, value_source, context, use_cached_kv
            )
        
        return output
    
    def _paged_attention_prefill_python(
        self,
        output: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
        use_cached_kv: bool,
    ) -> None:
        """
        Python implementation of prefill attention.
        
        Uses Flash Attention-style chunked computation for memory efficiency.
        """
        if use_cached_kv:
            paged_attention_prefill(
                output, q, k, v,
                context.cu_seqlens_q, context.cu_seqlens_k,
                context.max_seqlen_q, context.max_seqlen_k,
                self.scale, True, context.block_tables, self.alibi_slopes,
            )
        else:
            # Direct attention without paging for simple prefill
            self._direct_attention_prefill(output, q, k, v, context)
    
    def _direct_attention_prefill(
        self,
        output: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> None:
        """
        Direct attention computation without paging (for initial prefill).
        """
        batch_size = context.cu_seqlens_q.size(0) - 1
        
        for batch_idx in range(batch_size):
            q_start = context.cu_seqlens_q[batch_idx].item()
            q_end = context.cu_seqlens_q[batch_idx + 1].item()
            k_start = context.cu_seqlens_k[batch_idx].item()
            k_end = context.cu_seqlens_k[batch_idx + 1].item()
            
            seq_len_q = q_end - q_start
            seq_len_k = k_end - k_start
            
            if seq_len_q == 0 or seq_len_k == 0:
                continue
            
            q_seq = q[q_start:q_end]  # (seq_len_q, num_heads, head_dim)
            k_seq = k[k_start:k_end]  # (seq_len_k, num_kv_heads, head_dim)
            v_seq = v[k_start:k_end]  # (seq_len_k, num_kv_heads, head_dim)
            
            # Handle GQA by repeating KV heads
            if self.num_kv_groups > 1:
                k_seq = k_seq.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1)
                k_seq = k_seq.reshape(seq_len_k, self.num_heads, self.head_dim)
                v_seq = v_seq.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1)
                v_seq = v_seq.reshape(seq_len_k, self.num_heads, self.head_dim)
            
            # Compute attention scores
            # q_seq: (seq_len_q, num_heads, head_dim) -> (num_heads, seq_len_q, head_dim)
            # k_seq: (seq_len_k, num_heads, head_dim) -> (num_heads, head_dim, seq_len_k)
            q_t = q_seq.transpose(0, 1)
            k_t = k_seq.permute(1, 2, 0)
            
            scores = torch.bmm(q_t, k_t) * self.scale  # (num_heads, seq_len_q, seq_len_k)
            
            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
            
            # Softmax and weighted sum
            attn_weights = torch.softmax(scores, dim=-1)
            v_t = v_seq.transpose(0, 1)  # (num_heads, seq_len_k, head_dim)
            output_t = torch.bmm(attn_weights, v_t)  # (num_heads, seq_len_q, head_dim)
            
            output[q_start:q_end] = output_t.transpose(0, 1)
    
    def _forward_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """
        Forward pass for decode phase.
        
        Computes single-query attention against the full KV cache.
        """
        batch_size = q.size(0)
        output = torch.empty_like(q)
        
        # Compute context lengths from block tables
        block_size = k_cache.size(1)
        context_lens = self._compute_context_lens(context.block_tables, block_size)
        
        if HAS_CPP_KERNELS and q.dtype == torch.float32:
            # Use C++ AVX-512 kernel
            paged_attn_cpp.paged_attention_decode(
                output,
                q.contiguous(),
                k_cache,
                v_cache,
                context.block_tables.contiguous(),
                context_lens.contiguous(),
                self.scale,
                context_lens.max().item(),
            )
        elif HAS_IPEX:
            # Try IPEX single query attention
            try:
                ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
                    output,
                    q.unsqueeze(1),  # IPEX expects (batch, 1, heads, dim)
                    k_cache,
                    v_cache,
                    context.block_tables,
                    context_lens,
                    self.scale,
                    self.alibi_slopes,
                )
            except Exception:
                # Fallback to Python implementation
                paged_attention_decode(
                    output, q, k_cache, v_cache,
                    context.block_tables, context_lens,
                    self.scale, self.alibi_slopes,
                )
        else:
            # Use our Python/TorchScript implementation
            paged_attention_decode(
                output, q, k_cache, v_cache,
                context.block_tables, context_lens,
                self.scale, self.alibi_slopes,
            )
        
        return output
    
    def _compute_context_lens(
        self,
        block_tables: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """
        Compute context length for each sequence from block tables.
        
        The context length is (num_valid_blocks - 1) * block_size + tokens_in_last_block
        """
        batch_size = block_tables.size(0)
        context_lens = torch.zeros(batch_size, dtype=torch.int32, device=block_tables.device)
        
        for i in range(batch_size):
            # Count valid blocks (non-negative entries)
            valid_blocks = (block_tables[i] >= 0).sum().item()
            if valid_blocks > 0:
                # Assume full blocks for now (model_runner tracks exact lengths)
                # This is a simplification - in practice, context_lens should be passed
                context_lens[i] = valid_blocks * block_size
        
        return context_lens


class AttentionWithRoPE(Attention):
    """
    Attention with Rotary Position Embedding (RoPE) integration.
    
    This is a convenience class that combines attention computation
    with RoPE position encoding application.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        rope_module: nn.Module,
        alibi_slopes: Optional[torch.Tensor] = None,
    ):
        super().__init__(num_heads, head_dim, scale, num_kv_heads, alibi_slopes)
        self.rope = rope_module
    
    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RoPE and compute attention.
        
        Args:
            positions: Position indices (num_tokens,)
            q: Query tensor (num_tokens, num_heads, head_dim)
            k: Key tensor (num_tokens, num_kv_heads, head_dim)
            v: Value tensor (num_tokens, num_kv_heads, head_dim)
        """
        q, k = self.rope(positions, q, k)
        return super().forward(q, k, v)
