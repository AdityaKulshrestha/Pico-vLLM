"""
Attention Layer with PagedAttention for Intel Xeon CPUs

This module provides optimized attention computation using:
- Paged KV cache for memory-efficient long-context inference
- AVX-512 vectorized kernels (when C++ extension is available)
- Flash Attention-style chunked computation for prefill
- Optimized single-query attention for decode

Supports both standard Multi-Head Attention and Grouped Query Attention (GQA).
"""

import torch
from torch import nn
from typing import Optional

from picovllm.utils.context import get_context

# Import our custom PagedAttention kernels
from picovllm.kernels.paged_attention import (
    reshape_and_cache,
    paged_attention_prefill,
    paged_attention_decode,
    paged_attention_decode_chunked,
)

# Try to import C++ AVX-512 kernels for best performance
try:
    from picovllm.kernels.csrc import paged_attention_cpu as paged_attn_cpp
    HAS_CPP_KERNELS = True
except ImportError:
    HAS_CPP_KERNELS = False

# Try to import IPEX for additional Intel optimizations
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
    1. C++ AVX-512 kernel (fastest on Xeon)
    2. TorchScript optimized kernel (good performance)
    3. Pure Python loop (fallback)
    
    KV Cache Layout: (num_blocks, block_size, num_kv_heads, head_dim)
    The slot_mapping encodes both block_id and offset:
        block_id = slot // block_size
        offset = slot % block_size
    
    Args:
        key: Key tensor (N, num_kv_heads, head_dim)
        value: Value tensor (N, num_kv_heads, head_dim)
        k_cache: Key cache (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: Value cache (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: Slot indices for each token (N,)
        block_size: Number of tokens per cache block
    """
    if HAS_CPP_KERNELS and key.dtype == torch.float32:
        # Use AVX-512 optimized C++ kernel - fastest path
        paged_attn_cpp.reshape_and_cache(
            key.contiguous(),
            value.contiguous(),
            k_cache,
            v_cache,
            slot_mapping.contiguous(),
            block_size,
        )
    else:
        # Use TorchScript optimized kernel
        reshape_and_cache(key, value, k_cache, v_cache, slot_mapping, block_size)


class Attention(nn.Module):
    """
    Multi-Head Attention with PagedAttention support for Intel Xeon CPUs.
    
    Optimized using:
    - AVX-512 vectorized kernels for CPU (via C++ extension)
    - Flash Attention-style chunked computation for memory efficiency
    - Paged KV cache for efficient long-context inference
    - Support for Grouped Query Attention (GQA)
    
    The implementation automatically selects the best kernel based on:
    1. C++ AVX-512 kernel (if compiled and dtype is float32)
    2. IPEX kernel (if available)
    3. TorchScript kernel (fallback)
    
    Args:
        num_heads: Number of query attention heads
        head_dim: Dimension of each attention head
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        num_kv_heads: Number of key/value heads (for GQA, can be < num_heads)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        # Placeholder for KV cache (set by model_runner.allocate_kv_cache)
        self.k_cache: torch.Tensor = torch.tensor([])
        self.v_cache: torch.Tensor = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute attention with paged KV cache.
        
        Automatically dispatches to prefill or decode kernel based on context.
        
        Args:
            q: Query tensor (num_tokens, num_heads, head_dim)
            k: Key tensor (num_tokens, num_kv_heads, head_dim) 
            v: Value tensor (num_tokens, num_kv_heads, head_dim)
            dropout_p: Dropout probability (unused during inference)
            
        Returns:
            Output tensor (num_tokens, num_heads, head_dim)
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        block_size = k_cache.size(1) if k_cache.numel() > 0 else 256

        # Store KV to cache
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, block_size)

        if context.is_prefill:
            return self._forward_prefill(q, k, v, k_cache, v_cache, context, block_size)
        else:
            return self._forward_decode(q, k_cache, v_cache, context, block_size)
    
    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context,
        block_size: int,
    ) -> torch.Tensor:
        """
        Prefill phase: process entire prompt sequence.
        
        Uses Flash Attention-style chunked computation for long sequences
        to maintain O(N) memory complexity.
        """
        # Check if using prefix cache (cached KV from previous computation)
        use_cached_kv = context.block_tables is not None
        
        # Allocate output buffer
        output = torch.empty_like(q)
        
        if use_cached_kv and k_cache.numel() > 0:
            # Use paged attention with cached KV
            if HAS_CPP_KERNELS and q.dtype == torch.float32:
                # Best performance: C++ AVX-512 kernel
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
            elif HAS_IPEX:
                # Try IPEX kernel
                try:
                    ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
                        output, q, k_cache, v_cache,
                        context.cu_seqlens_q, context.cu_seqlens_k,
                        context.max_seqlen_q, context.max_seqlen_k,
                        self.scale, True, context.block_tables, alibi_slopes=None
                    )
                except Exception:
                    paged_attention_prefill(
                        output, q, k_cache, v_cache,
                        context.cu_seqlens_q, context.cu_seqlens_k,
                        context.max_seqlen_q, context.max_seqlen_k,
                        self.scale, True, context.block_tables, None
                    )
            else:
                paged_attention_prefill(
                    output, q, k_cache, v_cache,
                    context.cu_seqlens_q, context.cu_seqlens_k,
                    context.max_seqlen_q, context.max_seqlen_k,
                    self.scale, True, context.block_tables, None
                )
        else:
            # Direct attention (first prefill, no cache)
            self._direct_attention(output, q, k, v, context)
        
        return output
    
    def _direct_attention(
        self,
        output: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> None:
        """
        Direct attention computation without paging.
        
        Used for initial prefill when KV cache is empty.
        """
        batch_size = context.cu_seqlens_q.size(0) - 1
        
        for batch_idx in range(batch_size):
            q_start = context.cu_seqlens_q[batch_idx].item()
            q_end = context.cu_seqlens_q[batch_idx + 1].item()
            k_start = context.cu_seqlens_k[batch_idx].item() if context.cu_seqlens_k is not None else q_start
            k_end = context.cu_seqlens_k[batch_idx + 1].item() if context.cu_seqlens_k is not None else q_end
            
            seq_len_q = q_end - q_start
            seq_len_k = k_end - k_start
            
            if seq_len_q == 0 or seq_len_k == 0:
                continue
            
            q_seq = q[q_start:q_end]  # (seq_len_q, num_heads, head_dim)
            k_seq = k[k_start:k_end]  # (seq_len_k, num_kv_heads, head_dim)
            v_seq = v[k_start:k_end]  # (seq_len_k, num_kv_heads, head_dim)
            
            # Handle GQA by expanding KV heads
            if self.num_kv_groups > 1:
                k_seq = k_seq.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1)
                k_seq = k_seq.reshape(seq_len_k, self.num_heads, self.head_dim)
                v_seq = v_seq.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1)
                v_seq = v_seq.reshape(seq_len_k, self.num_heads, self.head_dim)
            
            # Compute attention: Q @ K^T -> softmax -> @ V
            # Transpose for batch matrix multiply
            q_t = q_seq.transpose(0, 1)  # (num_heads, seq_len_q, head_dim)
            k_t = k_seq.permute(1, 2, 0)  # (num_heads, head_dim, seq_len_k)
            
            scores = torch.bmm(q_t, k_t) * self.scale  # (num_heads, seq_len_q, seq_len_k)
            
            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
            
            # Softmax and output
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
        block_size: int,
    ) -> torch.Tensor:
        """
        Decode phase: generate one token at a time.
        
        Optimized for single-query attention against the full KV cache.
        """
        batch_size = q.size(0)
        output = torch.empty_like(q)
        
        # Compute context lengths from block tables
        context_lens = self._compute_context_lens(context.block_tables, block_size)
        
        if HAS_CPP_KERNELS and q.dtype == torch.float32:
            # Best performance: C++ AVX-512 kernel
            paged_attn_cpp.paged_attention_decode(
                output,
                q.contiguous(),
                k_cache,
                v_cache,
                context.block_tables.contiguous(),
                context_lens.contiguous(),
                self.scale,
                context_lens.max().item() if context_lens.numel() > 0 else 0,
            )
        elif HAS_IPEX:
            # Try IPEX kernel
            try:
                ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
                    output, q.unsqueeze(1), k_cache, v_cache,
                    context.block_tables, context_lens,
                    self.scale, None,
                )
            except Exception:
                paged_attention_decode(
                    output, q, k_cache, v_cache,
                    context.block_tables, context_lens,
                    self.scale, None,
                )
        else:
            # TorchScript kernel (with chunking for very long contexts)
            max_context = context_lens.max().item() if context_lens.numel() > 0 else 0
            if max_context > 4096:
                paged_attention_decode_chunked(
                    output, q, k_cache, v_cache,
                    context.block_tables, context_lens,
                    self.scale, None, kv_chunk_size=1024,
                )
            else:
                paged_attention_decode(
                    output, q, k_cache, v_cache,
                    context.block_tables, context_lens,
                    self.scale, None,
                )
        
        return output
    
    def _compute_context_lens(
        self,
        block_tables: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """
        Compute context length for each sequence from block tables.
        """
        if block_tables is None or block_tables.numel() == 0:
            return torch.tensor([], dtype=torch.int32)
        
        batch_size = block_tables.size(0)
        context_lens = torch.zeros(batch_size, dtype=torch.int32, device=block_tables.device)
        
        for i in range(batch_size):
            valid_blocks = (block_tables[i] >= 0).sum().item()
            context_lens[i] = valid_blocks * block_size
        
        return context_lens