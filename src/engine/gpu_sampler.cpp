#include "engine/gpu_sampler.hpp"

#include "common/cuda_check.hpp"

#include <algorithm>
#include <stdexcept>

namespace murmur {

namespace {

// The GPU sampler supports the CPU sampler's settings except an unlimited
// top_k (it keeps at most kMaxSamplingTopK candidates).
void Validate(const SamplingRequest& request) {
    const SamplingParams& p = request.params;
    const bool sampling = p.temperature > 0.0f;
    if (!(p.temperature >= 0.0f) || (sampling && (p.top_k < 1 || p.top_k > kMaxSamplingTopK)) ||
        !(p.top_p > 0.0f && p.top_p <= 1.0f) || !(p.repetition_penalty > 0.0f) ||
        !(request.unit_draw >= 0.0 && request.unit_draw < 1.0) || request.logits_row < -1) {
        throw std::invalid_argument("GpuSampler: sampling settings out of range");
    }
}

}  // namespace

GpuSampler::GpuSampler(int max_rows, int max_penalty_ids)
    : rows_host_(max_rows),
      rows_(max_rows),
      penalty_ids_host_(std::max(max_penalty_ids, 1)),
      penalty_ids_(std::max(max_penalty_ids, 1)),
      tokens_(max_rows),
      tokens_host_(max_rows) {}

// Samples one token per request from the rows of `logits` (GPU memory, rows
// `logits_stride` apart; ids [0, candidate_count) are the candidates):
// request r's own row r unless it names another (each row at most once),
// queued on `stream`. Blocks until the tokens are back; the returned span
// stays valid until the next call. The logits are modified.
std::span<const TokenId> GpuSampler::Sample(float* logits, int logits_stride, int candidate_count,
                                            std::span<const SamplingRequest> requests, cudaStream_t stream) {
    if (requests.size() > rows_host_.Size() || candidate_count < kMaxSamplingTopK) {
        throw std::invalid_argument("GpuSampler::Sample: too many rows or too few candidates");
    }
    // The previous call's copies have finished: it waited for its tokens.
    std::size_t penalty_count = 0;
    for (std::size_t r = 0; r < requests.size(); ++r) {
        const SamplingRequest& request = requests[r];
        Validate(request);
        if (penalty_count + request.penalty_ids.size() > penalty_ids_host_.Size()) {
            throw std::invalid_argument("GpuSampler::Sample: too many penalty ids");
        }
        std::ranges::copy(request.penalty_ids, penalty_ids_host_.Data() + penalty_count);
        rows_host_.Data()[r] = SamplingRow{.temperature = request.params.temperature,
                                           .top_k = request.params.top_k,
                                           .top_p = request.params.top_p,
                                           .repetition_penalty = request.params.repetition_penalty,
                                           .unit_draw = request.unit_draw,
                                           .penalty_begin = static_cast<int>(penalty_count),
                                           .penalty_end = static_cast<int>(penalty_count + request.penalty_ids.size()),
                                           .logits_row = request.logits_row >= 0 ? request.logits_row : static_cast<int>(r),
                                           .allowed = request.allowed};
        penalty_count += request.penalty_ids.size();
    }
    const int rows = static_cast<int>(requests.size());
    CudaCheck(cudaMemcpyAsync(rows_.Data(), rows_host_.Data(), rows * sizeof(SamplingRow), cudaMemcpyHostToDevice,
                              stream));
    CudaCheck(cudaMemcpyAsync(penalty_ids_.Data(), penalty_ids_host_.Data(), penalty_count * sizeof(TokenId),
                              cudaMemcpyHostToDevice, stream));
    SampleRows(logits, logits_stride, candidate_count, rows_.Data(), penalty_ids_.Data(), rows, tokens_.Data(), stream);
    CudaCheck(cudaMemcpyAsync(tokens_host_.Data(), tokens_.Data(), rows * sizeof(TokenId), cudaMemcpyDeviceToHost,
                              stream));
    CudaCheck(cudaStreamSynchronize(stream));
    return {tokens_host_.Data(), requests.size()};
}

}  // namespace murmur
