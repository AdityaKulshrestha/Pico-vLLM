"""
Unit tests for PagedAttention kernels.

Run with:
    pytest picovllm/kernels/test_paged_attention.py -v
"""

import torch
import pytest
import math
from typing import Tuple

from picovllm.kernels.paged_attention import (
    reshape_and_cache,
    paged_attention_prefill,
    paged_attention_decode,
    PagedAttention,
)


def create_test_tensors(
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, ...]:
    """Create test tensors for attention computation."""
    total_tokens = batch_size * seq_len
    num_blocks = (total_tokens + block_size - 1) // block_size + 10  # Extra blocks
    
    query = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype)
    key = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype)
    value = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype)
    
    key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype)
    value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype)
    
    # Create cumulative sequence lengths
    cu_seqlens_q = torch.tensor([i * seq_len for i in range(batch_size + 1)], dtype=torch.int32)
    cu_seqlens_k = cu_seqlens_q.clone()
    
    # Create slot mapping (sequential slots)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int32)
    
    # Create block tables
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    block_tables = torch.zeros(batch_size, blocks_per_seq, dtype=torch.int32)
    for b in range(batch_size):
        for i in range(blocks_per_seq):
            block_tables[b, i] = b * blocks_per_seq + i
    
    return (query, key, value, key_cache, value_cache, 
            cu_seqlens_q, cu_seqlens_k, slot_mapping, block_tables)


class TestReshapeAndCache:
    """Tests for KV cache reshape operation."""
    
    def test_basic_cache_store(self):
        """Test basic key-value storage into cache."""
        num_tokens = 32
        num_kv_heads = 4
        head_dim = 64
        block_size = 16
        num_blocks = 4
        
        key = torch.randn(num_tokens, num_kv_heads, head_dim)
        value = torch.randn(num_tokens, num_kv_heads, head_dim)
        key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32)
        
        reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, block_size)
        
        # Verify storage
        for i in range(num_tokens):
            block_id = i // block_size
            offset = i % block_size
            assert torch.allclose(key_cache[block_id, offset], key[i])
            assert torch.allclose(value_cache[block_id, offset], value[i])
    
    def test_non_sequential_slots(self):
        """Test storage with non-sequential slot mapping."""
        num_tokens = 16
        num_kv_heads = 2
        head_dim = 32
        block_size = 8
        num_blocks = 4
        
        key = torch.randn(num_tokens, num_kv_heads, head_dim)
        value = torch.randn(num_tokens, num_kv_heads, head_dim)
        key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        
        # Reverse slot mapping
        slot_mapping = torch.arange(num_tokens - 1, -1, -1, dtype=torch.int32)
        
        reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, block_size)
        
        # Verify storage
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            block_id = slot // block_size
            offset = slot % block_size
            assert torch.allclose(key_cache[block_id, offset], key[i])
    
    def test_invalid_slots_skipped(self):
        """Test that invalid slots (-1) are skipped."""
        num_tokens = 16
        num_kv_heads = 2
        head_dim = 32
        block_size = 8
        num_blocks = 2
        
        key = torch.randn(num_tokens, num_kv_heads, head_dim)
        value = torch.randn(num_tokens, num_kv_heads, head_dim)
        key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        
        # Mix valid and invalid slots
        slot_mapping = torch.tensor([0, -1, 1, -1, 2, 3, -1, 4, 5, 6, 7, -1, 8, 9, 10, 11], dtype=torch.int32)
        
        reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, block_size)
        
        # Verify valid slots were stored
        valid_indices = [0, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15]
        for i in valid_indices:
            slot = slot_mapping[i].item()
            if slot >= 0:
                block_id = slot // block_size
                offset = slot % block_size
                assert torch.allclose(key_cache[block_id, offset], key[i])


class TestPrefillAttention:
    """Tests for prefill attention computation."""
    
    def test_single_sequence(self):
        """Test attention with single sequence."""
        seq_len = 64
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        block_size = 16
        num_blocks = 8
        
        query = torch.randn(seq_len, num_heads, head_dim)
        key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32)
        cu_seqlens_k = cu_seqlens_q.clone()
        
        blocks_per_seq = (seq_len + block_size - 1) // block_size
        block_tables = torch.arange(blocks_per_seq, dtype=torch.int32).unsqueeze(0)
        
        output = torch.empty_like(query)
        scale = head_dim ** -0.5
        
        paged_attention_prefill(
            output, query, key_cache, value_cache,
            cu_seqlens_q, cu_seqlens_k,
            seq_len, seq_len, scale, True, block_tables, None
        )
        
        assert output.shape == query.shape
        assert not torch.isnan(output).any()
    
    def test_batch_attention(self):
        """Test attention with multiple sequences."""
        batch_size = 3
        seq_lens = [32, 48, 64]
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        block_size = 16
        
        total_tokens = sum(seq_lens)
        max_seq_len = max(seq_lens)
        num_blocks = (total_tokens + block_size - 1) // block_size + 10
        
        query = torch.randn(total_tokens, num_heads, head_dim)
        key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        
        cu_seqlens_q = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(batch_size)], dtype=torch.int32)
        cu_seqlens_k = cu_seqlens_q.clone()
        
        # Create block tables
        max_blocks = (max_seq_len + block_size - 1) // block_size
        block_tables = torch.full((batch_size, max_blocks), -1, dtype=torch.int32)
        block_id = 0
        for b, seq_len in enumerate(seq_lens):
            blocks_needed = (seq_len + block_size - 1) // block_size
            for i in range(blocks_needed):
                block_tables[b, i] = block_id
                block_id += 1
        
        output = torch.empty_like(query)
        scale = head_dim ** -0.5
        
        paged_attention_prefill(
            output, query, key_cache, value_cache,
            cu_seqlens_q, cu_seqlens_k,
            max_seq_len, max_seq_len, scale, True, block_tables, None
        )
        
        assert output.shape == query.shape
        assert not torch.isnan(output).any()


class TestDecodeAttention:
    """Tests for decode attention computation."""
    
    def test_single_query(self):
        """Test decode attention with single query."""
        batch_size = 1
        context_len = 128
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        block_size = 16
        num_blocks = 16
        
        query = torch.randn(batch_size, num_heads, head_dim)
        key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        
        blocks_per_seq = (context_len + block_size - 1) // block_size
        block_tables = torch.arange(blocks_per_seq, dtype=torch.int32).unsqueeze(0)
        context_lens = torch.tensor([context_len], dtype=torch.int32)
        
        output = torch.empty_like(query)
        scale = head_dim ** -0.5
        
        paged_attention_decode(
            output, query, key_cache, value_cache,
            block_tables, context_lens, scale, None
        )
        
        assert output.shape == query.shape
        assert not torch.isnan(output).any()
    
    def test_batch_decode(self):
        """Test decode attention with batch of queries."""
        batch_size = 4
        context_lens_list = [64, 128, 96, 256]
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        block_size = 32
        num_blocks = 64
        
        query = torch.randn(batch_size, num_heads, head_dim)
        key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        
        max_context = max(context_lens_list)
        max_blocks = (max_context + block_size - 1) // block_size
        block_tables = torch.full((batch_size, max_blocks), -1, dtype=torch.int32)
        block_id = 0
        for b, ctx_len in enumerate(context_lens_list):
            blocks_needed = (ctx_len + block_size - 1) // block_size
            for i in range(blocks_needed):
                block_tables[b, i] = block_id
                block_id += 1
        
        context_lens = torch.tensor(context_lens_list, dtype=torch.int32)
        
        output = torch.empty_like(query)
        scale = head_dim ** -0.5
        
        paged_attention_decode(
            output, query, key_cache, value_cache,
            block_tables, context_lens, scale, None
        )
        
        assert output.shape == query.shape
        assert not torch.isnan(output).any()


class TestPagedAttentionModule:
    """Tests for the high-level PagedAttention module."""
    
    def test_module_creation(self):
        """Test module instantiation."""
        module = PagedAttention(
            num_heads=8,
            head_dim=64,
            num_kv_heads=2,
            scale=0.125,
            block_size=256,
        )
        
        assert module.num_heads == 8
        assert module.head_dim == 64
        assert module.num_kv_heads == 2
        assert module.num_kv_groups == 4
    
    def test_forward_prefill(self):
        """Test prefill forward pass."""
        module = PagedAttention(
            num_heads=4,
            head_dim=32,
            num_kv_heads=2,
            scale=32 ** -0.5,
            block_size=16,
        )
        
        seq_len = 64
        num_blocks = 8
        
        query = torch.randn(seq_len, 4, 32)
        key_cache = torch.randn(num_blocks, 16, 2, 32)
        value_cache = torch.randn(num_blocks, 16, 2, 32)
        
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32)
        cu_seqlens_k = cu_seqlens_q.clone()
        block_tables = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        
        output = module.forward_prefill(
            query, key_cache, value_cache,
            cu_seqlens_q, cu_seqlens_k,
            seq_len, seq_len, True, block_tables
        )
        
        assert output.shape == query.shape
    
    def test_forward_decode(self):
        """Test decode forward pass."""
        module = PagedAttention(
            num_heads=4,
            head_dim=32,
            num_kv_heads=2,
            scale=32 ** -0.5,
            block_size=16,
        )
        
        batch_size = 2
        context_len = 64
        num_blocks = 16
        
        query = torch.randn(batch_size, 4, 32)
        key_cache = torch.randn(num_blocks, 16, 2, 32)
        value_cache = torch.randn(num_blocks, 16, 2, 32)
        
        block_tables = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
        context_lens = torch.tensor([context_len, context_len], dtype=torch.int32)
        
        output = module.forward_decode(
            query, key_cache, value_cache,
            block_tables, context_lens
        )
        
        assert output.shape == query.shape


class TestNumericalAccuracy:
    """Tests for numerical correctness of attention computation."""
    
    def test_attention_against_reference(self):
        """Compare PagedAttention output against reference implementation."""
        seq_len = 32
        num_heads = 4
        num_kv_heads = 4  # No GQA for simpler comparison
        head_dim = 32
        
        query = torch.randn(seq_len, num_heads, head_dim)
        key = torch.randn(seq_len, num_kv_heads, head_dim)
        value = torch.randn(seq_len, num_kv_heads, head_dim)
        
        scale = head_dim ** -0.5
        
        # Reference implementation (standard attention)
        q_t = query.transpose(0, 1)  # (heads, seq, dim)
        k_t = key.permute(1, 2, 0)   # (heads, dim, seq)
        
        scores = torch.bmm(q_t, k_t) * scale
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        v_t = value.transpose(0, 1)
        ref_output = torch.bmm(attn_weights, v_t).transpose(0, 1)
        
        # PagedAttention implementation
        block_size = 16
        num_blocks = (seq_len + block_size - 1) // block_size
        key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        
        # Store KV in cache
        slot_mapping = torch.arange(seq_len, dtype=torch.int32)
        reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, block_size)
        
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
        block_tables = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
        
        paged_output = torch.empty_like(query)
        paged_attention_prefill(
            paged_output, query, key_cache, value_cache,
            cu_seqlens, cu_seqlens, seq_len, seq_len,
            scale, True, block_tables, None
        )
        
        # Compare outputs (allow some numerical tolerance)
        assert torch.allclose(paged_output, ref_output, rtol=1e-3, atol=1e-4), \
            f"Max diff: {(paged_output - ref_output).abs().max()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
