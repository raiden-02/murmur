#pragma once

#include "common/device_buffer.hpp"
#include "common/pinned_buffer.hpp"
#include "engine/kernels/sampling.cuh"
#include "engine/sampler.hpp"
#include "model/tokenizer.hpp"

#include <cuda_runtime.h>

#include <span>

namespace murmur {

// One row's sampling request: its settings, its random draw, and the
// distinct tokens already in its conversation (for the repetition penalty).
struct SamplingRequest {
    SamplingParams params;
    double unit_draw = 0.0;
    std::span<const TokenId> penalty_ids;
    int logits_row = -1;   // the logits row to sample (-1: the request's own index)
    const std::uint32_t* allowed = nullptr;   // GPU bitmask of allowed tokens (TokenGrammar::DeviceMask), or all
};

// Picks the next token of many conversations at once on the GPU, straight
// from the logits there: only the chosen ids come back to the CPU (4 bytes
// each instead of 600 KB of logits).
class GpuSampler {
public:
    GpuSampler(int max_rows, int max_penalty_ids);

    std::span<const TokenId> Sample(float* logits, int logits_stride, int candidate_count,
                                    std::span<const SamplingRequest> requests, cudaStream_t stream);

private:
    PinnedBuffer<SamplingRow> rows_host_;
    DeviceBuffer<SamplingRow> rows_;
    PinnedBuffer<TokenId> penalty_ids_host_;
    DeviceBuffer<TokenId> penalty_ids_;
    DeviceBuffer<TokenId> tokens_;
    PinnedBuffer<TokenId> tokens_host_;
};

}  // namespace murmur
