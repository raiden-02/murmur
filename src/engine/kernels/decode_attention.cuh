#pragma once

#include "common/device_buffer.hpp"
#include "engine/kernels/paged_kv.cuh"

#include <cuda_runtime.h>

namespace murmur {

struct DecodeAttentionShape {
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int max_context;   // longest sequence
    int max_batch;     // most sequences per call
};

// Partial results of the split attention, allocated once per model.
class DecodeAttentionWorkspace {
public:
    explicit DecodeAttentionWorkspace(const DecodeAttentionShape& shape);

    const DecodeAttentionShape& Shape() const { return shape_; }
    float* PartialOutputs() { return partial_outputs_.Data(); }
    float* PartialMaxes() { return partial_maxes_.Data(); }
    float* PartialSums() { return partial_sums_.Data(); }

private:
    DecodeAttentionShape shape_;
    DeviceBuffer<float> partial_outputs_;   // max_batch x num_heads x max splits x head_dim
    DeviceBuffer<float> partial_maxes_;     // max_batch x num_heads x max splits
    DeviceBuffer<float> partial_sums_;      // max_batch x num_heads x max splits
};

void DecodeAttention(const float* q, const KvLayer& cache, const PagedKv& kv, const int* context_lengths, int batch,
                     float* out, DecodeAttentionWorkspace& workspace, cudaStream_t stream);

}  // namespace murmur
