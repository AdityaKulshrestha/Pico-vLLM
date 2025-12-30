/*
 * PagedAttention C++ Kernels for Intel Xeon CPUs
 * 
 * This module provides high-performance AVX-512 optimized kernels for:
 * - KV cache reshape operations
 * - Single-query decode attention
 * - Vectorized softmax computation
 * 
 * Compilation:
 *   g++ -O3 -march=native -fopenmp -mavx512f -mavx512bw -shared -fPIC \
 *       -I<python_include> -I<torch_include> paged_attention_cpu.cpp \
 *       -o paged_attention_cpu.so
 * 
 * Or via PyTorch's CppExtension in setup.py
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>

// Alignment for AVX-512 (64 bytes = 512 bits)
#define AVX512_ALIGNMENT 64

namespace paged_attention {

// ==============================================================================
// AVX-512 Utility Functions
// ==============================================================================

// Horizontal sum of __m512 (16 floats)
inline float _mm512_reduce_add_ps(__m512 v) {
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(low, high);
    __m128 low128 = _mm256_castps256_ps128(sum256);
    __m128 high128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// Horizontal max of __m512
inline float _mm512_reduce_max_ps_custom(__m512 v) {
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm512_extractf32x8_ps(v, 1);
    __m256 max256 = _mm256_max_ps(low, high);
    __m128 low128 = _mm256_castps256_ps128(max256);
    __m128 high128 = _mm256_extractf128_ps(max256, 1);
    __m128 max128 = _mm_max_ps(low128, high128);
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max128);
}

// Fast exponential approximation using AVX-512
// Based on Schraudolph's algorithm with improved accuracy
inline __m512 _mm512_exp_ps_fast(__m512 x) {
    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 ln2 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 c0 = _mm512_set1_ps(0.5f);
    const __m512 c1 = _mm512_set1_ps(0.166666666666666f);
    const __m512 c2 = _mm512_set1_ps(0.041666666666666f);
    
    // Clamp input to avoid overflow/underflow
    const __m512 max_val = _mm512_set1_ps(88.0f);
    const __m512 min_val = _mm512_set1_ps(-88.0f);
    x = _mm512_max_ps(_mm512_min_ps(x, max_val), min_val);
    
    // Convert to 2^(x * log2(e))
    __m512 fx = _mm512_mul_ps(x, log2e);
    
    // Round to nearest integer
    __m512 rx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    
    // Compute fractional part
    __m512 f = _mm512_sub_ps(fx, rx);
    
    // Polynomial approximation for 2^f where f is in [-0.5, 0.5]
    __m512 f2 = _mm512_mul_ps(f, f);
    __m512 f3 = _mm512_mul_ps(f2, f);
    __m512 result = _mm512_add_ps(one, f);
    result = _mm512_fmadd_ps(c0, f2, result);
    result = _mm512_fmadd_ps(c1, f3, result);
    
    // Scale by 2^n using integer manipulation
    __m512i n = _mm512_cvtps_epi32(rx);
    n = _mm512_add_epi32(n, _mm512_set1_epi32(127));
    n = _mm512_slli_epi32(n, 23);
    __m512 scale = _mm512_castsi512_ps(n);
    
    return _mm512_mul_ps(result, scale);
}

// ==============================================================================
// KV Cache Operations
// ==============================================================================

void reshape_and_cache_cpu(
    torch::Tensor& key,           // (num_tokens, num_kv_heads, head_dim)
    torch::Tensor& value,         // (num_tokens, num_kv_heads, head_dim)
    torch::Tensor& key_cache,     // (num_blocks, block_size, num_kv_heads, head_dim)
    torch::Tensor& value_cache,   // (num_blocks, block_size, num_kv_heads, head_dim)
    torch::Tensor& slot_mapping,  // (num_tokens,)
    int64_t block_size
) {
    const int64_t num_tokens = key.size(0);
    const int64_t num_kv_heads = key.size(1);
    const int64_t head_dim = key.size(2);
    const int64_t kv_size = num_kv_heads * head_dim;
    
    // Get raw pointers
    float* key_ptr = key.data_ptr<float>();
    float* value_ptr = value.data_ptr<float>();
    float* key_cache_ptr = key_cache.data_ptr<float>();
    float* value_cache_ptr = value_cache.data_ptr<float>();
    int32_t* slot_ptr = slot_mapping.data_ptr<int32_t>();
    
    // Parallel copy with OpenMP
    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t i = 0; i < num_tokens; ++i) {
        int32_t slot = slot_ptr[i];
        if (slot < 0) continue;
        
        float* src_k = key_ptr + i * kv_size;
        float* src_v = value_ptr + i * kv_size;
        float* dst_k = key_cache_ptr + slot * kv_size;
        float* dst_v = value_cache_ptr + slot * kv_size;
        
        // Vectorized copy using AVX-512
        int64_t j = 0;
        for (; j + 16 <= kv_size; j += 16) {
            __m512 k_vec = _mm512_loadu_ps(src_k + j);
            __m512 v_vec = _mm512_loadu_ps(src_v + j);
            _mm512_storeu_ps(dst_k + j, k_vec);
            _mm512_storeu_ps(dst_v + j, v_vec);
        }
        // Handle remainder
        for (; j < kv_size; ++j) {
            dst_k[j] = src_k[j];
            dst_v[j] = src_v[j];
        }
    }
}

// ==============================================================================
// Single Query Decode Attention (AVX-512 Optimized)
// ==============================================================================

void single_query_attention_avx512(
    float* output,           // (num_heads, head_dim)
    const float* query,      // (num_heads, head_dim)
    const float* keys,       // (context_len, num_heads, head_dim)
    const float* values,     // (context_len, num_heads, head_dim)
    int64_t num_heads,
    int64_t head_dim,
    int64_t context_len,
    float scale
) {
    // Process each head in parallel
    #pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < num_heads; ++h) {
        const float* q = query + h * head_dim;
        float* out = output + h * head_dim;
        
        // Allocate aligned buffer for attention scores
        std::vector<float, std::aligned_allocator<float, AVX512_ALIGNMENT>> scores(context_len);
        
        // Compute attention scores: q @ K^T
        float max_score = -std::numeric_limits<float>::infinity();
        
        for (int64_t t = 0; t < context_len; ++t) {
            const float* k = keys + t * num_heads * head_dim + h * head_dim;
            
            // Vectorized dot product
            __m512 sum_vec = _mm512_setzero_ps();
            int64_t d = 0;
            
            for (; d + 16 <= head_dim; d += 16) {
                __m512 q_vec = _mm512_loadu_ps(q + d);
                __m512 k_vec = _mm512_loadu_ps(k + d);
                sum_vec = _mm512_fmadd_ps(q_vec, k_vec, sum_vec);
            }
            
            float dot = _mm512_reduce_add_ps(sum_vec);
            
            // Handle remainder
            for (; d < head_dim; ++d) {
                dot += q[d] * k[d];
            }
            
            scores[t] = dot * scale;
            max_score = std::max(max_score, scores[t]);
        }
        
        // Compute softmax: exp(score - max) / sum(exp)
        __m512 max_vec = _mm512_set1_ps(max_score);
        __m512 sum_vec = _mm512_setzero_ps();
        
        int64_t t = 0;
        for (; t + 16 <= context_len; t += 16) {
            __m512 score_vec = _mm512_load_ps(&scores[t]);
            score_vec = _mm512_sub_ps(score_vec, max_vec);
            __m512 exp_vec = _mm512_exp_ps_fast(score_vec);
            _mm512_store_ps(&scores[t], exp_vec);
            sum_vec = _mm512_add_ps(sum_vec, exp_vec);
        }
        
        float sum_exp = _mm512_reduce_add_ps(sum_vec);
        
        // Handle remainder
        for (; t < context_len; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        
        // Normalize
        float inv_sum = 1.0f / sum_exp;
        __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
        
        t = 0;
        for (; t + 16 <= context_len; t += 16) {
            __m512 score_vec = _mm512_load_ps(&scores[t]);
            score_vec = _mm512_mul_ps(score_vec, inv_sum_vec);
            _mm512_store_ps(&scores[t], score_vec);
        }
        for (; t < context_len; ++t) {
            scores[t] *= inv_sum;
        }
        
        // Compute output: attn_weights @ V
        // Initialize output to zero
        std::memset(out, 0, head_dim * sizeof(float));
        
        for (t = 0; t < context_len; ++t) {
            const float* v = values + t * num_heads * head_dim + h * head_dim;
            float w = scores[t];
            __m512 w_vec = _mm512_set1_ps(w);
            
            int64_t d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 out_vec = _mm512_loadu_ps(out + d);
                __m512 v_vec = _mm512_loadu_ps(v + d);
                out_vec = _mm512_fmadd_ps(w_vec, v_vec, out_vec);
                _mm512_storeu_ps(out + d, out_vec);
            }
            for (; d < head_dim; ++d) {
                out[d] += w * v[d];
            }
        }
    }
}

void paged_attention_decode_cpu(
    torch::Tensor& output,        // (batch_size, num_heads, head_dim)
    torch::Tensor& query,         // (batch_size, num_heads, head_dim)
    torch::Tensor& key_cache,     // (num_blocks, block_size, num_kv_heads, head_dim)
    torch::Tensor& value_cache,   // (num_blocks, block_size, num_kv_heads, head_dim)
    torch::Tensor& block_tables,  // (batch_size, max_blocks)
    torch::Tensor& context_lens,  // (batch_size,)
    float scale,
    int64_t max_context_len
) {
    const int64_t batch_size = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t head_dim = query.size(2);
    const int64_t num_kv_heads = key_cache.size(2);
    const int64_t block_size = key_cache.size(1);
    const int64_t num_kv_groups = num_heads / num_kv_heads;
    
    float* output_ptr = output.data_ptr<float>();
    float* query_ptr = query.data_ptr<float>();
    float* key_cache_ptr = key_cache.data_ptr<float>();
    float* value_cache_ptr = value_cache.data_ptr<float>();
    int32_t* block_tables_ptr = block_tables.data_ptr<int32_t>();
    int32_t* context_lens_ptr = context_lens.data_ptr<int32_t>();
    
    const int64_t max_blocks = block_tables.size(1);
    const int64_t kv_head_stride = head_dim;
    const int64_t kv_block_stride = block_size * num_kv_heads * head_dim;
    
    // Process each sequence in batch
    #pragma omp parallel for schedule(dynamic)
    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t context_len = context_lens_ptr[b];
        if (context_len == 0) {
            std::memset(output_ptr + b * num_heads * head_dim, 0, 
                       num_heads * head_dim * sizeof(float));
            continue;
        }
        
        float* q = query_ptr + b * num_heads * head_dim;
        float* out = output_ptr + b * num_heads * head_dim;
        int32_t* block_table = block_tables_ptr + b * max_blocks;
        
        // Allocate temporary buffers for gathered KV
        const int64_t kv_size = context_len * num_heads * head_dim;
        std::vector<float> keys(kv_size);
        std::vector<float> values(kv_size);
        
        // Gather KV from paged cache
        int64_t offset = 0;
        int64_t tokens_remaining = context_len;
        
        for (int64_t block_idx = 0; tokens_remaining > 0; ++block_idx) {
            int32_t block_id = block_table[block_idx];
            if (block_id < 0) break;
            
            int64_t tokens_in_block = std::min(block_size, tokens_remaining);
            
            // Copy KV with GQA expansion
            for (int64_t t = 0; t < tokens_in_block; ++t) {
                for (int64_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                    const float* src_k = key_cache_ptr + block_id * kv_block_stride + 
                                        t * num_kv_heads * head_dim + kv_h * head_dim;
                    const float* src_v = value_cache_ptr + block_id * kv_block_stride + 
                                        t * num_kv_heads * head_dim + kv_h * head_dim;
                    
                    // Expand to all heads in the group
                    for (int64_t g = 0; g < num_kv_groups; ++g) {
                        int64_t h = kv_h * num_kv_groups + g;
                        float* dst_k = keys.data() + (offset + t) * num_heads * head_dim + h * head_dim;
                        float* dst_v = values.data() + (offset + t) * num_heads * head_dim + h * head_dim;
                        
                        std::memcpy(dst_k, src_k, head_dim * sizeof(float));
                        std::memcpy(dst_v, src_v, head_dim * sizeof(float));
                    }
                }
            }
            
            offset += tokens_in_block;
            tokens_remaining -= tokens_in_block;
        }
        
        // Compute attention with AVX-512
        single_query_attention_avx512(
            out, q, keys.data(), values.data(),
            num_heads, head_dim, context_len, scale
        );
    }
}

// ==============================================================================
// Prefill Attention (Chunked Flash Attention Style)
// ==============================================================================

void flash_attention_chunk_cpu(
    float* output,           // (chunk_size, num_heads, head_dim)
    const float* query,      // (chunk_size, num_heads, head_dim) 
    const float* keys,       // (seq_len_k, num_heads, head_dim)
    const float* values,     // (seq_len_k, num_heads, head_dim)
    int64_t chunk_size,
    int64_t seq_len_k,
    int64_t num_heads,
    int64_t head_dim,
    float scale,
    bool causal,
    int64_t chunk_start      // Starting position in sequence
) {
    // Process each head in parallel
    #pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < num_heads; ++h) {
        // Process each query position
        for (int64_t q_idx = 0; q_idx < chunk_size; ++q_idx) {
            const float* q = query + q_idx * num_heads * head_dim + h * head_dim;
            float* out = output + q_idx * num_heads * head_dim + h * head_dim;
            
            int64_t q_pos = chunk_start + q_idx;
            int64_t k_end = causal ? std::min(q_pos + 1, seq_len_k) : seq_len_k;
            
            if (k_end == 0) {
                std::memset(out, 0, head_dim * sizeof(float));
                continue;
            }
            
            // Allocate score buffer
            std::vector<float> scores(k_end);
            
            // Compute attention scores
            float max_score = -std::numeric_limits<float>::infinity();
            
            for (int64_t k_idx = 0; k_idx < k_end; ++k_idx) {
                const float* k = keys + k_idx * num_heads * head_dim + h * head_dim;
                
                // Vectorized dot product
                __m512 sum_vec = _mm512_setzero_ps();
                int64_t d = 0;
                
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 q_vec = _mm512_loadu_ps(q + d);
                    __m512 k_vec = _mm512_loadu_ps(k + d);
                    sum_vec = _mm512_fmadd_ps(q_vec, k_vec, sum_vec);
                }
                
                float dot = _mm512_reduce_add_ps(sum_vec);
                for (; d < head_dim; ++d) {
                    dot += q[d] * k[d];
                }
                
                scores[k_idx] = dot * scale;
                max_score = std::max(max_score, scores[k_idx]);
            }
            
            // Softmax
            float sum_exp = 0.0f;
            for (int64_t k_idx = 0; k_idx < k_end; ++k_idx) {
                scores[k_idx] = std::exp(scores[k_idx] - max_score);
                sum_exp += scores[k_idx];
            }
            
            float inv_sum = 1.0f / sum_exp;
            for (int64_t k_idx = 0; k_idx < k_end; ++k_idx) {
                scores[k_idx] *= inv_sum;
            }
            
            // Weighted sum of values
            std::memset(out, 0, head_dim * sizeof(float));
            
            for (int64_t k_idx = 0; k_idx < k_end; ++k_idx) {
                const float* v = values + k_idx * num_heads * head_dim + h * head_dim;
                float w = scores[k_idx];
                __m512 w_vec = _mm512_set1_ps(w);
                
                int64_t d = 0;
                for (; d + 16 <= head_dim; d += 16) {
                    __m512 out_vec = _mm512_loadu_ps(out + d);
                    __m512 v_vec = _mm512_loadu_ps(v + d);
                    out_vec = _mm512_fmadd_ps(w_vec, v_vec, out_vec);
                    _mm512_storeu_ps(out + d, out_vec);
                }
                for (; d < head_dim; ++d) {
                    out[d] += w * v[d];
                }
            }
        }
    }
}

void paged_attention_prefill_cpu(
    torch::Tensor& output,        // (num_tokens, num_heads, head_dim)
    torch::Tensor& query,         // (num_tokens, num_heads, head_dim)
    torch::Tensor& key_cache,     // (num_blocks, block_size, num_kv_heads, head_dim)
    torch::Tensor& value_cache,   // (num_blocks, block_size, num_kv_heads, head_dim)
    torch::Tensor& cu_seqlens_q,  // (batch_size + 1,)
    torch::Tensor& cu_seqlens_k,  // (batch_size + 1,)
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float scale,
    bool causal,
    std::optional<torch::Tensor> block_tables,
    int64_t chunk_size = 256
) {
    const int64_t batch_size = cu_seqlens_q.size(0) - 1;
    const int64_t num_heads = query.size(1);
    const int64_t head_dim = query.size(2);
    const int64_t num_kv_heads = key_cache.size(2);
    const int64_t block_size = key_cache.size(1);
    const int64_t num_kv_groups = num_heads / num_kv_heads;
    
    float* output_ptr = output.data_ptr<float>();
    float* query_ptr = query.data_ptr<float>();
    float* key_cache_ptr = key_cache.data_ptr<float>();
    float* value_cache_ptr = value_cache.data_ptr<float>();
    int32_t* cu_seqlens_q_ptr = cu_seqlens_q.data_ptr<int32_t>();
    int32_t* cu_seqlens_k_ptr = cu_seqlens_k.data_ptr<int32_t>();
    
    int32_t* block_tables_ptr = block_tables.has_value() ? 
                                 block_tables->data_ptr<int32_t>() : nullptr;
    int64_t max_blocks = block_tables.has_value() ? block_tables->size(1) : 0;
    
    const int64_t kv_block_stride = block_size * num_kv_heads * head_dim;
    
    // Process each sequence
    #pragma omp parallel for schedule(dynamic)
    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t q_start = cu_seqlens_q_ptr[b];
        int64_t q_end = cu_seqlens_q_ptr[b + 1];
        int64_t k_start = cu_seqlens_k_ptr[b];
        int64_t k_end = cu_seqlens_k_ptr[b + 1];
        
        int64_t seq_len_q = q_end - q_start;
        int64_t seq_len_k = k_end - k_start;
        
        if (seq_len_q == 0 || seq_len_k == 0) continue;
        
        // Gather KV from cache with GQA expansion
        std::vector<float> keys(seq_len_k * num_heads * head_dim);
        std::vector<float> values(seq_len_k * num_heads * head_dim);
        
        int64_t offset = 0;
        int64_t tokens_remaining = seq_len_k;
        int64_t num_blocks_needed = (seq_len_k + block_size - 1) / block_size;
        
        for (int64_t block_idx = 0; block_idx < num_blocks_needed; ++block_idx) {
            int64_t block_id;
            if (block_tables_ptr) {
                block_id = block_tables_ptr[b * max_blocks + block_idx];
            } else {
                block_id = block_idx;  // Sequential blocks for simple prefill
            }
            
            if (block_id < 0) break;
            
            int64_t tokens_in_block = std::min(block_size, tokens_remaining);
            
            for (int64_t t = 0; t < tokens_in_block; ++t) {
                for (int64_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
                    const float* src_k = key_cache_ptr + block_id * kv_block_stride + 
                                        t * num_kv_heads * head_dim + kv_h * head_dim;
                    const float* src_v = value_cache_ptr + block_id * kv_block_stride + 
                                        t * num_kv_heads * head_dim + kv_h * head_dim;
                    
                    for (int64_t g = 0; g < num_kv_groups; ++g) {
                        int64_t h = kv_h * num_kv_groups + g;
                        float* dst_k = keys.data() + (offset + t) * num_heads * head_dim + h * head_dim;
                        float* dst_v = values.data() + (offset + t) * num_heads * head_dim + h * head_dim;
                        
                        std::memcpy(dst_k, src_k, head_dim * sizeof(float));
                        std::memcpy(dst_v, src_v, head_dim * sizeof(float));
                    }
                }
            }
            
            offset += tokens_in_block;
            tokens_remaining -= tokens_in_block;
        }
        
        // Compute attention in chunks
        float* q_ptr_batch = query_ptr + q_start * num_heads * head_dim;
        float* out_ptr_batch = output_ptr + q_start * num_heads * head_dim;
        
        for (int64_t chunk_start_idx = 0; chunk_start_idx < seq_len_q; chunk_start_idx += chunk_size) {
            int64_t chunk_end_idx = std::min(chunk_start_idx + chunk_size, seq_len_q);
            int64_t actual_chunk_size = chunk_end_idx - chunk_start_idx;
            
            flash_attention_chunk_cpu(
                out_ptr_batch + chunk_start_idx * num_heads * head_dim,
                q_ptr_batch + chunk_start_idx * num_heads * head_dim,
                keys.data(),
                values.data(),
                actual_chunk_size,
                seq_len_k,
                num_heads,
                head_dim,
                scale,
                causal,
                chunk_start_idx
            );
        }
    }
}

// ==============================================================================
// Python Bindings
// ==============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reshape_and_cache", &reshape_and_cache_cpu, 
          "Reshape and cache KV pairs (CPU AVX-512)");
    m.def("paged_attention_decode", &paged_attention_decode_cpu,
          "PagedAttention decode phase (CPU AVX-512)");
    m.def("paged_attention_prefill", &paged_attention_prefill_cpu,
          "PagedAttention prefill phase (CPU AVX-512)");
}

// Custom allocator for aligned memory
template <typename T, size_t Alignment>
class aligned_allocator {
public:
    using value_type = T;
    
    aligned_allocator() noexcept = default;
    
    template <typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}
    
    T* allocate(std::size_t n) {
        void* ptr = std::aligned_alloc(Alignment, n * sizeof(T));
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr, std::size_t) noexcept {
        std::free(ptr);
    }
};

} // namespace paged_attention
