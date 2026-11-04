#pragma once

#include <cuda_runtime.h>

namespace murmur::lab {

// How one position's next-token distribution under a candidate model
// compares with the reference model's.
struct PositionQuality {
    float kl;                // KL(reference || candidate), in nats
    float nll_reference;     // -log p(actual next token), reference; 0 when there is none
    float nll_candidate;     // the same under the candidate
    int same_top_token;      // 1 when both rank the same token first
};

void CompareLogits(const float* reference, const float* candidate, int rows, int row_stride, int vocab,
                   const int* next_tokens, PositionQuality* out, cudaStream_t stream);

}  // namespace murmur::lab
