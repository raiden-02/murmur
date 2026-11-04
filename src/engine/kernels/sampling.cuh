#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace murmur {

// Most candidates the GPU sampler keeps per row (its top_k limit).
inline constexpr int kMaxSamplingTopK = 32;

// How one row of logits is turned into a token, in GPU memory.
struct SamplingRow {
    float temperature;          // 0 = greedy: always the top token
    int top_k;                  // 1..kMaxSamplingTopK candidates (when sampling)
    float top_p;                // then the fewest of those covering this probability mass
    float repetition_penalty;   // applied to the row's earlier tokens
    double unit_draw;           // the random number in [0, 1) that picks among the kept candidates
    int penalty_begin;          // the row's distinct earlier tokens: penalty_ids[begin, end)
    int penalty_end;
    int logits_row;             // which row of the logits to sample from
    const std::uint32_t* allowed;   // GPU bitmask of the tokens that may be picked (bit id % 32 of word
                                    // id / 32), or null for all: constrained decoding
};

void SampleRows(float* logits, int logits_stride, int candidate_count, const SamplingRow* rows,
                const std::int32_t* penalty_ids, int row_count, std::int32_t* out, cudaStream_t stream);

}  // namespace murmur
