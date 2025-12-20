import torch
from torch import nn
# import triton
# import triton.language as tl

# from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import intel_extension_for_pytorch as ipex
from picovllm.utils.context import get_context

# NOTE - Why not using direct libxsmm - manually tile the matrices, manage the cache hierarchies and run softmax using AVX512.
# NOTE - oneDNN solves these by managing the cache hierarchies internally, performs fusion on the matrix multiplicaiton and softmax
# @triton.jit
# def store_kvcache_kernel(
#     key_ptr,
#     key_stride,
#     value_ptr,
#     value_stride,
#     k_cache_ptr,
#     v_cache_ptr,
#     slot_mapping_ptr,
#     D: tl.constexpr,
# ): 
#     idx = tl.program_id(0)    # Token ID
#     slot = tl.load(slot_mapping_ptr + idx) 
#     if slot == -1: return
#     key_offsets = idx * key_stride + tl.arange(0, D)
#     value_offsets = idx * value_stride + tl.arange(0, D)

#     key = tl.load(key_ptr + key_offsets)
#     value = tl.load(value_ptr + value_offsets)

#     cache_offsets = slot * D + tl.arange(0, D)

#     tl.store(k_cache_ptr + cache_offsets, key)
#     tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # On CPU, simple advanced indexing is often faster than custom
    # SYCL kernel because it leverages OpenMP and AVX-512 automatically.
    # breakpoint()
    # KV cache is in shape of (Total blocks, block_size, num_heads, head_dim)
    # Key, Value are in shape of (N, num_heads, head_dim)
    # Flatten the last two dimension, choose the block id using slot_mapping and store the key, value
    # NOTE - The slot map contains the actual index + offset encoded: 257 = 257 // block_size (index) + 257 % block_size (offset)
    # NOTE - key is contiguous but the value is not contiguous in memory. 


    # NOTE - Look for alternate to make it efficient
    for i in range(N):
        kv_index = slot_mapping[i] // 256 # Block size
        kv_offset = slot_mapping[i] % 256
        if kv_index == -1:
            continue
        k_cache[kv_index, kv_offset, :, :] = key[i].reshape(num_heads, head_dim)
        v_cache[kv_index, kv_offset, :, :] = value[i].reshape(num_heads, head_dim)

    # k_cache[slot_mapping] = key.reshape(key.size(0), -1)
    # v_cache[slot_mapping] = value.reshape(value.size(0), -1)
    # store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float = 0.0) -> torch.Tensor:
        context = get_context()
        print("This si the shape: ", q.shape, k.shape, v.shape)
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:            # Prefix cache
                k, v = k_cache, v_cache

            # o = flash_attn_varlen_func(
                # q, k, v,
                # max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                # max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                # softmax_scale=self.scale, causal=True, block_table=context.block_table)
            
            with torch.cpu.amp.autocast():
                o = torch.ops.torch_ipex.flash_attn_varlen(
                    q, k, v,
                    context.max_seqlen_q, context.cu_seqlens_q,
                    context.max_seqlen_k, context.cu_seqlens_k,
                    dropout_p, self.scale, True
                )
            
        
        # else:
        #     o = flash_attn_with_kvcache(
        #         q.unsqueeze(1), k_cache, v_cache,
        #         cache_seqlens=context.context_lens, block_table=context.block_tables,
        #         softmax_scale=self.scale, causal=True
        #     )

        #     out = torch.empty(1, num_heads, head_size)

        #     torch.ops.torch_ipex.single_query_cached_kv_attention(
        #         out, q.unsqueeze(1), k_cache, v_cache
            # )
        
        return o