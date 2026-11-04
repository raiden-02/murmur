#pragma once

#include "engine/kernels/paged_kv.cuh"

#include <cuda_runtime.h>

#include <vector>

namespace murmur {

std::vector<float> RopeInverseFrequencies(int head_dim, float theta);

struct QkvLayout {
    int num_heads;      // query heads
    int num_kv_heads;   // key/value heads
    int head_dim;
};

void RopeAndCacheQkv(const float* qkv, int rows, const int* positions, const int* cache_rows,
                     const float* inverse_frequencies, const QkvLayout& layout, float* q, const KvLayer& cache,
                     cudaStream_t stream);

}  // namespace murmur
