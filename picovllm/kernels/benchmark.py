"""
Benchmark script for PagedAttention kernels on Intel Xeon CPUs.

Tests performance of:
- KV cache reshape operations
- Prefill attention
- Decode attention

Run with:
    python -m picovllm.kernels.benchmark

Environment variables:
    OMP_NUM_THREADS: Number of OpenMP threads
    KMP_AFFINITY: Thread affinity settings
"""

import os
import time
import torch
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Set optimal environment variables for Intel CPUs
os.environ.setdefault('KMP_BLOCKTIME', '0')
os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')

from picovllm.kernels.paged_attention import (
    reshape_and_cache,
    paged_attention_prefill,
    paged_attention_decode,
)

# Try to import C++ kernels
try:
    from picovllm.kernels.csrc import paged_attention_cpu as cpp_kernels
    HAS_CPP_KERNELS = True
except ImportError:
    HAS_CPP_KERNELS = False
    print("C++ kernels not available, using Python implementation")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    batch_size: int = 1
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    block_size: int = 256
    num_blocks: int = 1024
    seq_len: int = 2048
    context_len: int = 4096
    dtype: torch.dtype = torch.float32
    num_warmup: int = 3
    num_iterations: int = 10


def benchmark_reshape_and_cache(config: BenchmarkConfig) -> Dict[str, float]:
    """Benchmark KV cache reshape operations."""
    num_tokens = config.seq_len
    
    # Create tensors
    key = torch.randn(num_tokens, config.num_kv_heads, config.head_dim, dtype=config.dtype)
    value = torch.randn(num_tokens, config.num_kv_heads, config.head_dim, dtype=config.dtype)
    key_cache = torch.zeros(config.num_blocks, config.block_size, config.num_kv_heads, config.head_dim, dtype=config.dtype)
    value_cache = torch.zeros(config.num_blocks, config.block_size, config.num_kv_heads, config.head_dim, dtype=config.dtype)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32)
    
    results = {}
    
    # Benchmark Python implementation
    for _ in range(config.num_warmup):
        reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, config.block_size)
    
    start = time.perf_counter()
    for _ in range(config.num_iterations):
        reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, config.block_size)
    end = time.perf_counter()
    
    python_time = (end - start) / config.num_iterations * 1000  # ms
    results['python_ms'] = python_time
    
    # Benchmark C++ implementation if available
    if HAS_CPP_KERNELS:
        key_cache.zero_()
        value_cache.zero_()
        
        for _ in range(config.num_warmup):
            cpp_kernels.reshape_and_cache(
                key.contiguous(), value.contiguous(),
                key_cache, value_cache,
                slot_mapping.contiguous(), config.block_size
            )
        
        start = time.perf_counter()
        for _ in range(config.num_iterations):
            cpp_kernels.reshape_and_cache(
                key.contiguous(), value.contiguous(),
                key_cache, value_cache,
                slot_mapping.contiguous(), config.block_size
            )
        end = time.perf_counter()
        
        cpp_time = (end - start) / config.num_iterations * 1000
        results['cpp_ms'] = cpp_time
        results['speedup'] = python_time / cpp_time
    
    # Compute throughput
    bytes_transferred = num_tokens * config.num_kv_heads * config.head_dim * 4 * 2  # K and V, float32
    results['throughput_gbps'] = bytes_transferred / (results.get('cpp_ms', python_time) / 1000) / 1e9
    
    return results


def benchmark_prefill(config: BenchmarkConfig) -> Dict[str, float]:
    """Benchmark prefill attention."""
    batch_size = config.batch_size
    seq_len = config.seq_len
    total_tokens = batch_size * seq_len
    
    # Create tensors
    query = torch.randn(total_tokens, config.num_heads, config.head_dim, dtype=config.dtype)
    key_cache = torch.randn(config.num_blocks, config.block_size, config.num_kv_heads, config.head_dim, dtype=config.dtype)
    value_cache = torch.randn(config.num_blocks, config.block_size, config.num_kv_heads, config.head_dim, dtype=config.dtype)
    output = torch.empty_like(query)
    
    # Create cumulative sequence lengths
    cu_seqlens_q = torch.tensor([i * seq_len for i in range(batch_size + 1)], dtype=torch.int32)
    cu_seqlens_k = cu_seqlens_q.clone()
    
    # Create block tables (assuming sequential blocks)
    blocks_per_seq = (seq_len + config.block_size - 1) // config.block_size
    block_tables = torch.arange(batch_size * blocks_per_seq, dtype=torch.int32).view(batch_size, blocks_per_seq)
    
    scale = config.head_dim ** -0.5
    
    results = {}
    
    # Benchmark Python implementation
    for _ in range(config.num_warmup):
        paged_attention_prefill(
            output, query, key_cache, value_cache,
            cu_seqlens_q, cu_seqlens_k,
            seq_len, seq_len, scale, True, block_tables, None
        )
    
    start = time.perf_counter()
    for _ in range(config.num_iterations):
        paged_attention_prefill(
            output, query, key_cache, value_cache,
            cu_seqlens_q, cu_seqlens_k,
            seq_len, seq_len, scale, True, block_tables, None
        )
    end = time.perf_counter()
    
    python_time = (end - start) / config.num_iterations * 1000
    results['python_ms'] = python_time
    
    # Compute FLOPS
    # Attention: 2 * batch * heads * seq_q * seq_k * head_dim (QK^T and softmax@V)
    flops = 2 * batch_size * config.num_heads * seq_len * seq_len * config.head_dim
    results['tflops'] = flops / (python_time / 1000) / 1e12
    
    return results


def benchmark_decode(config: BenchmarkConfig) -> Dict[str, float]:
    """Benchmark decode attention."""
    batch_size = config.batch_size
    context_len = config.context_len
    
    # Create tensors
    query = torch.randn(batch_size, config.num_heads, config.head_dim, dtype=config.dtype)
    key_cache = torch.randn(config.num_blocks, config.block_size, config.num_kv_heads, config.head_dim, dtype=config.dtype)
    value_cache = torch.randn(config.num_blocks, config.block_size, config.num_kv_heads, config.head_dim, dtype=config.dtype)
    output = torch.empty_like(query)
    
    # Create block tables
    blocks_per_seq = (context_len + config.block_size - 1) // config.block_size
    block_tables = torch.zeros(batch_size, blocks_per_seq, dtype=torch.int32)
    for b in range(batch_size):
        for i in range(blocks_per_seq):
            block_tables[b, i] = b * blocks_per_seq + i
    
    context_lens = torch.full((batch_size,), context_len, dtype=torch.int32)
    scale = config.head_dim ** -0.5
    
    results = {}
    
    # Benchmark Python implementation
    for _ in range(config.num_warmup):
        paged_attention_decode(
            output, query, key_cache, value_cache,
            block_tables, context_lens, scale, None
        )
    
    start = time.perf_counter()
    for _ in range(config.num_iterations):
        paged_attention_decode(
            output, query, key_cache, value_cache,
            block_tables, context_lens, scale, None
        )
    end = time.perf_counter()
    
    python_time = (end - start) / config.num_iterations * 1000
    results['python_ms'] = python_time
    
    # Benchmark C++ implementation if available
    if HAS_CPP_KERNELS:
        output.zero_()
        
        for _ in range(config.num_warmup):
            cpp_kernels.paged_attention_decode(
                output, query.contiguous(),
                key_cache, value_cache,
                block_tables.contiguous(), context_lens.contiguous(),
                scale, context_len
            )
        
        start = time.perf_counter()
        for _ in range(config.num_iterations):
            cpp_kernels.paged_attention_decode(
                output, query.contiguous(),
                key_cache, value_cache,
                block_tables.contiguous(), context_lens.contiguous(),
                scale, context_len
            )
        end = time.perf_counter()
        
        cpp_time = (end - start) / config.num_iterations * 1000
        results['cpp_ms'] = cpp_time
        results['speedup'] = python_time / cpp_time
    
    # Compute FLOPS
    flops = 2 * batch_size * config.num_heads * context_len * config.head_dim
    best_time = results.get('cpp_ms', python_time)
    results['tflops'] = flops / (best_time / 1000) / 1e12
    
    # Compute memory bandwidth
    # Read: Q (batch * heads * dim) + K,V (batch * context * heads * dim * 2)
    # Write: O (batch * heads * dim)
    bytes_read = (batch_size * config.num_heads * config.head_dim +
                  batch_size * context_len * config.num_heads * config.head_dim * 2) * 4
    bytes_written = batch_size * config.num_heads * config.head_dim * 4
    results['bandwidth_gbps'] = (bytes_read + bytes_written) / (best_time / 1000) / 1e9
    
    return results


def print_results(name: str, results: Dict[str, float]):
    """Print benchmark results."""
    print(f"\n{name}:")
    print("-" * 50)
    for key, value in results.items():
        if 'ms' in key:
            print(f"  {key}: {value:.3f} ms")
        elif 'speedup' in key:
            print(f"  {key}: {value:.2f}x")
        elif 'tflops' in key:
            print(f"  {key}: {value:.3f} TFLOPS")
        elif 'gbps' in key or 'bandwidth' in key:
            print(f"  {key}: {value:.2f} GB/s")
        else:
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark PagedAttention kernels')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-heads', type=int, default=32, help='Number of query heads')
    parser.add_argument('--num-kv-heads', type=int, default=8, help='Number of KV heads')
    parser.add_argument('--head-dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--seq-len', type=int, default=2048, help='Sequence length for prefill')
    parser.add_argument('--context-len', type=int, default=4096, help='Context length for decode')
    parser.add_argument('--block-size', type=int, default=256, help='KV cache block size')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        seq_len=args.seq_len,
        context_len=args.context_len,
        block_size=args.block_size,
        num_iterations=args.iterations,
    )
    
    print("=" * 60)
    print("PagedAttention Benchmark for Intel Xeon CPUs")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Query heads: {config.num_heads}, KV heads: {config.num_kv_heads}")
    print(f"  Head dimension: {config.head_dim}")
    print(f"  Sequence length (prefill): {config.seq_len}")
    print(f"  Context length (decode): {config.context_len}")
    print(f"  Block size: {config.block_size}")
    print(f"  C++ kernels available: {HAS_CPP_KERNELS}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    
    # Run benchmarks
    print_results("KV Cache Reshape", benchmark_reshape_and_cache(config))
    print_results("Prefill Attention", benchmark_prefill(config))
    print_results("Decode Attention", benchmark_decode(config))
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
