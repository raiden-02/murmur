#include "lab/logit_comparison.cuh"

#include "common/cuda_check.hpp"
#include "engine/kernels/block_reduce.cuh"

#include <cstddef>

namespace murmur::lab {

namespace {

constexpr int kThreads = 512;

__device__ float BlockMin(float value) {
    return BlockReduce(value, FLT_MAX, [](float v) {
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            v = fminf(v, __shfl_xor_sync(kFullWarpMask, v, offset));
        }
        return v;
    });
}

// The first index holding the row's maximum (ids below 2^24 are exact as floats).
__device__ int FirstArgmax(const float* row, int vocab, float max_value) {
    float first = FLT_MAX;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (row[i] == max_value) {
            first = fminf(first, static_cast<float>(i));
        }
    }
    return static_cast<int>(BlockMin(first));
}

// One block per position: both log-softmaxes (max-subtracted), then
// KL = sum_i p_i (log p_i - log q_i) over the reference distribution p.
__global__ void CompareLogitsKernel(const float* reference, const float* candidate, int row_stride, int vocab,
                                    const int* next_tokens, PositionQuality* out) {
    const float* p_logits = reference + static_cast<std::size_t>(blockIdx.x) * row_stride;
    const float* q_logits = candidate + static_cast<std::size_t>(blockIdx.x) * row_stride;

    float p_max = -FLT_MAX;
    float q_max = -FLT_MAX;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        p_max = fmaxf(p_max, p_logits[i]);
        q_max = fmaxf(q_max, q_logits[i]);
    }
    p_max = BlockMax(p_max);
    q_max = BlockMax(q_max);

    float p_sum = 0.0f;
    float q_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        p_sum += expf(p_logits[i] - p_max);
        q_sum += expf(q_logits[i] - q_max);
    }
    const float p_log_total = p_max + logf(BlockSum(p_sum));
    const float q_log_total = q_max + logf(BlockSum(q_sum));

    float kl = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float log_p = p_logits[i] - p_log_total;
        kl += expf(log_p) * (log_p - (q_logits[i] - q_log_total));
    }
    kl = BlockSum(kl);
    const int p_top = FirstArgmax(p_logits, vocab, p_max);
    const int q_top = FirstArgmax(q_logits, vocab, q_max);

    if (threadIdx.x == 0) {
        const int next = next_tokens[blockIdx.x];
        out[blockIdx.x] = {.kl = fmaxf(kl, 0.0f),
                           .nll_reference = next >= 0 ? p_log_total - p_logits[next] : 0.0f,
                           .nll_candidate = next >= 0 ? q_log_total - q_logits[next] : 0.0f,
                           .same_top_token = p_top == q_top ? 1 : 0};
    }
}

}  // namespace

// Compares `rows` positions' logits (row r at reference + r * row_stride,
// the first `vocab` entries) under two models. next_tokens[r] (GPU memory)
// is the token that actually follows position r, or -1 for none.
void CompareLogits(const float* reference, const float* candidate, int rows, int row_stride, int vocab,
                   const int* next_tokens, PositionQuality* out, cudaStream_t stream) {
    if (rows <= 0) {
        return;
    }
    CompareLogitsKernel<<<rows, kThreads, 0, stream>>>(reference, candidate, row_stride, vocab, next_tokens, out);
    CheckKernelLaunch();
}

}  // namespace murmur::lab
