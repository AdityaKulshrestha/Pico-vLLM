"""
Build script for PagedAttention CPU kernels.

Build with:
    python setup.py build_ext --inplace

Or install with:
    pip install -e .
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def get_extensions():
    extensions = []
    
    # Check for AVX-512 support
    extra_compile_args = [
        '-O3',
        '-fopenmp',
        '-std=c++17',
    ]
    
    # Try to detect CPU features
    try:
        import subprocess
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        cpu_info = result.stdout.lower()
        
        if 'avx512' in cpu_info:
            extra_compile_args.extend([
                '-mavx512f',
                '-mavx512bw',
                '-mavx512dq',
                '-mavx512vl',
            ])
            print("AVX-512 support detected, enabling AVX-512 optimizations")
        elif 'avx2' in cpu_info:
            extra_compile_args.extend([
                '-mavx2',
                '-mfma',
            ])
            print("AVX2 support detected, enabling AVX2 optimizations")
        else:
            extra_compile_args.append('-march=native')
            print("Using native CPU optimizations")
    except Exception:
        extra_compile_args.append('-march=native')
    
    extra_link_args = [
        '-fopenmp',
    ]
    
    paged_attention_ext = CppExtension(
        name='picovllm.kernels.csrc.paged_attention_cpu',
        sources=[
            'picovllm/kernels/csrc/paged_attention_cpu.cpp',
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    extensions.append(paged_attention_ext)
    
    return extensions


setup(
    name='paged_attention_cpu',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
)
