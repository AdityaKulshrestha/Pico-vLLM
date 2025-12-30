# C++ PagedAttention Kernels

This directory contains high-performance C++ kernels for PagedAttention optimized for Intel Xeon CPUs.

## Features

- **AVX-512 Vectorization**: Uses Intel AVX-512 intrinsics for maximum SIMD throughput
- **OpenMP Parallelization**: Multi-threaded execution across heads and batch dimension
- **Cache-Optimized**: Memory access patterns designed for CPU cache hierarchy
- **Fast Exponential**: Custom AVX-512 exponential approximation for softmax

## Building

### Option 1: In-place build

```bash
cd picovllm/kernels/csrc
python setup.py build_ext --inplace
```

### Option 2: Install via pip

From the Pico-vLLM root directory:

```bash
pip install -e .
```

## Requirements

- GCC 9+ or Clang 10+
- PyTorch 2.0+
- CPU with AVX-512 support (Intel Xeon Skylake-SP or newer)

## Usage

The C++ kernels are automatically used when available:

```python
from picovllm.kernels.paged_attention import (
    reshape_and_cache,
    paged_attention_prefill,
    paged_attention_decode,
)

# If C++ kernels are compiled, they will be used automatically
# for float32 tensors when calling these functions
```

## Environment Variables

- `OMP_NUM_THREADS`: Number of OpenMP threads (default: number of CPU cores)
- `KMP_AFFINITY`: Thread affinity settings (recommended: `granularity=fine,compact,1,0`)
- `KMP_BLOCKTIME`: Time to wait before sleeping (recommended: `0`)

## Performance Tips

1. Set optimal environment variables:
   ```bash
   export OMP_NUM_THREADS=$(nproc)
   export KMP_AFFINITY="granularity=fine,compact,1,0"
   export KMP_BLOCKTIME=0
   ```

2. Use BF16 or FP32 for best performance on Intel CPUs

3. Ensure tensors are contiguous before passing to kernels

## Architecture

### KV Cache Reshape (`reshape_and_cache`)
- Parallel scatter of K/V tensors to paged cache
- AVX-512 vectorized memory copy
- Handles non-contiguous slot mappings

### Prefill Attention (`paged_attention_prefill`)
- Flash Attention-style chunked computation
- Parallel across batch and heads
- Fused softmax with numerical stability

### Decode Attention (`paged_attention_decode`)
- Single-query attention against full KV cache
- Block-sparse memory access via block tables
- Online softmax for very long contexts
