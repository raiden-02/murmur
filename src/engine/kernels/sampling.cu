#include "engine/kernels/sampling.cuh"

#include "common/cuda_check.hpp"
#include "engine/kernels/block_reduce.cuh"
#include "engine/kernels/launch.cuh"

#include <cmath>
#include <cstddef>

namespace murmur {

namespace {

constexpr int kThreadsPerBlock = 512;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

struct Candidate {
    float logit;
    std::int32_t id;
};

// Sorting order of the candidates: the higher logit first, the lower id on a
// tie (the CPU sampler's order, and torch's).
__device__ __forceinline__ bool Before(Candidate a, Candidate b) {
    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
}

__device__ __forceinline__ Candidate WarpFirst(Candidate best) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const Candidate other{__shfl_xor_sync(kFullWarpMask, best.logit, offset),
                              __shfl_xor_sync(kFullWarpMask, best.id, offset)};
        best = Before(other, best) ? other : best;
    }
    return best;
}

// The block's first candidate in sorting order; every thread receives it.
__device__ Candidate BlockFirst(Candidate best) {
    __shared__ Candidate warp_best[kWarpsPerBlock];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    best = WarpFirst(best);
    __syncthreads();   // warp_best may still be read from the previous call
    if (lane == 0) {
        warp_best[warp] = best;
    }
    __syncthreads();
    best = lane < kWarpsPerBlock ? warp_best[lane] : Candidate{-INFINITY, INT32_MAX};
    return WarpFirst(best);
}

// One block per row of logits, in Hugging Face's order:
//   0. tokens the row's mask forbids are never candidates (constrained decoding)
//   1. repetition penalty on the row's earlier tokens (distinct ids, so no
//      two threads touch the same logit)
//   2. greedy: the first candidate overall; otherwise
//   3. top-k: each thread keeps its own k best (sorted, in registers and
//      local memory), then k rounds of "best head of any thread's list"
//      merge them into the row's k best
//   4. temperature, softmax over the k, top-p and the weighted draw, on one
//      thread (k is at most 32)
// Temperature scales every logit the same way, so it doesn't change which k
// are best and is only applied to them.
__global__ void SampleRowsKernel(float* logits, int logits_stride, int candidate_count, const SamplingRow* rows,
                                 const std::int32_t* penalty_ids, std::int32_t* out) {
    __shared__ Candidate top[kMaxSamplingTopK];
    const SamplingRow params = rows[blockIdx.x];
    float* row = logits + static_cast<std::size_t>(params.logits_row) * logits_stride;

    if (params.repetition_penalty != 1.0f) {
        for (int i = params.penalty_begin + threadIdx.x; i < params.penalty_end; i += blockDim.x) {
            const std::int32_t id = penalty_ids[i];
            if (id >= 0 && id < candidate_count) {
                const float v = row[id];
                row[id] = v < 0.0f ? v * params.repetition_penalty : v / params.repetition_penalty;
            }
        }
        __syncthreads();
    }

    const int k = params.temperature == 0.0f ? 1 : params.top_k;
    Candidate local[kMaxSamplingTopK];
#pragma unroll
    for (int j = 0; j < kMaxSamplingTopK; ++j) {
        local[j] = {-INFINITY, INT32_MAX};
    }
    for (int i = threadIdx.x; i < candidate_count; i += blockDim.x) {
        if (params.allowed != nullptr && ((params.allowed[i >> 5] >> (i & 31)) & 1u) == 0) {
            continue;   // the grammar forbids this token here
        }
        const Candidate c{row[i], i};
        if (!Before(c, local[k - 1])) {
            continue;   // the common case once the list is full: not among this thread's best
        }
        int j = k - 1;
        while (j > 0 && Before(c, local[j - 1])) {
            local[j] = local[j - 1];
            --j;
        }
        local[j] = c;
    }

    int head = 0;
    for (int round = 0; round < k; ++round) {
        const Candidate mine = head < k ? local[head] : Candidate{-INFINITY, INT32_MAX};
        const Candidate best = BlockFirst(mine);
        if (mine.id == best.id && mine.logit == best.logit) {
            ++head;   // ids are unique, so exactly one thread owned it
        }
        if (threadIdx.x == 0) {
            top[round] = best;
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    if (k == 1) {
        out[blockIdx.x] = top[0].id;
        return;
    }
    double probabilities[kMaxSamplingTopK];
    const float max_logit = top[0].logit / params.temperature;
    double sum = 0.0;
    for (int j = 0; j < k; ++j) {
        probabilities[j] = exp(static_cast<double>(top[j].logit / params.temperature) - max_logit);
        sum += probabilities[j];
    }
    for (int j = 0; j < k; ++j) {
        probabilities[j] /= sum;
    }
    // Nucleus: walking from the least likely, drop candidates while the
    // dropped mass stays within 1 - top_p (Hugging Face's rule).
    int keep = k;
    double dropped = 0.0;
    while (keep > 1 && dropped + probabilities[keep - 1] <= 1.0 - params.top_p) {
        dropped += probabilities[--keep];
    }
    double kept_mass = 0.0;
    for (int j = 0; j < keep; ++j) {
        kept_mass += probabilities[j];
    }
    double draw = params.unit_draw * kept_mass;
    std::int32_t choice = top[keep - 1].id;
    for (int j = 0; j + 1 < keep; ++j) {
        draw -= probabilities[j];
        if (draw < 0.0) {
            choice = top[j].id;
            break;
        }
    }
    out[blockIdx.x] = choice;
}

}  // namespace

// Picks one token per entry of `rows` on the GPU: entry r samples logits
// row rows[r].logits_row (rows are `logits_stride` floats apart; only ids
// [0, candidate_count) are considered; each row at most once), with the same
// steps and result as the CPU Sampler given the same unit draw. The logits
// are modified in place by the penalty. `rows`,
// `penalty_ids` and `out` (one id per row) live in GPU memory.
void SampleRows(float* logits, int logits_stride, int candidate_count, const SamplingRow* rows,
                const std::int32_t* penalty_ids, int row_count, std::int32_t* out, cudaStream_t stream) {
    RequireArgument(row_count >= 0 && candidate_count > 0 && logits_stride >= candidate_count,
                    "SampleRows: bad shape");
    if (row_count == 0) {
        return;
    }
    SampleRowsKernel<<<row_count, kThreadsPerBlock, 0, stream>>>(logits, logits_stride, candidate_count, rows,
                                                                 penalty_ids, out);
    CheckKernelLaunch();
}

}  // namespace murmur
