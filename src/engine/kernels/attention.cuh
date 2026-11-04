#pragma once

#include "engine/kernels/paged_kv.cuh"

#include <cuda_runtime.h>

namespace murmur {

struct AttentionShape {
    int num_heads;
    int num_kv_heads;
    int head_dim;
};

// Where each query row sits: its position in its sequence (it attends to
// positions 0..position) and which row of the page tables is its sequence's.
// Both arrays live in GPU memory, one entry per query row.
struct QueryRows {
    const int* positions;
    const int* sequences;
};

int MaxAttentionContext(int head_dim);

void CausalAttention(const float* q, const KvLayer& cache, const PagedKv& kv, const QueryRows& rows, int row_count,
                     int max_context, float* out, const AttentionShape& shape, cudaStream_t stream);
void FlashAttention(const float* q, const KvLayer& cache, const PagedKv& kv, const QueryRows& rows, int row_count,
                    float* out, const AttentionShape& shape, cudaStream_t stream);

}  // namespace murmur
