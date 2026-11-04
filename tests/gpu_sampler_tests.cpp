#include "common/cuda_stream.hpp"
#include "common/device_buffer.hpp"
#include "engine/gpu_sampler.hpp"
#include "engine/sampler.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace murmur {
namespace {

constexpr int kVocab = 5000;
constexpr int kStride = 5003;   // rows needn't be packed

std::vector<float> RandomLogits(std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 3.0f);
    std::vector<float> logits(kVocab);
    for (float& logit : logits) {
        logit = dist(rng);
    }
    return logits;
}

std::vector<TokenId> Distinct(const std::vector<TokenId>& history) {
    const std::set<TokenId> unique(history.begin(), history.end());
    return {unique.begin(), unique.end()};
}

TEST(GpuSampler, PicksExactlyWhatTheCpuSamplerPicksForTheSameDraw) {
    const SamplingParams settings[] = {
        {.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 1.0f},     // greedy
        {.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 1.3f},     // greedy, penalized
        {.temperature = 0.7f, .top_k = 20, .top_p = 0.8f, .repetition_penalty = 1.1f},    // the model's defaults
        {.temperature = 1.0f, .top_k = 1, .top_p = 1.0f, .repetition_penalty = 1.0f},
        {.temperature = 1.5f, .top_k = 32, .top_p = 1.0f, .repetition_penalty = 1.0f},
        {.temperature = 0.3f, .top_k = 7, .top_p = 0.5f, .repetition_penalty = 2.0f},
    };
    std::mt19937 rng(5);
    std::vector<std::vector<float>> rows;
    std::vector<std::vector<TokenId>> histories;
    std::vector<SamplingRequest> requests;
    for (const SamplingParams& params : settings) {
        for (const double draw : {0.0, 0.31, 0.77, 0.9999}) {
            rows.push_back(RandomLogits(rng));
            std::vector<TokenId> history;
            for (int i = 0; i < 40; ++i) {
                history.push_back(static_cast<TokenId>(rng() % 60));   // repeats on purpose
            }
            histories.push_back(history);
            requests.push_back({.params = params, .unit_draw = draw});
        }
    }
    std::vector<std::vector<TokenId>> distinct;
    for (const auto& history : histories) {
        distinct.push_back(Distinct(history));
    }
    std::vector<float> packed(rows.size() * kStride, -1.0f);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        std::ranges::copy(rows[r], packed.begin() + r * kStride);
        requests[r].penalty_ids = distinct[r];
    }
    DeviceBuffer<float> logits(packed.size());
    logits.CopyFromHost(packed);
    CudaStream stream;
    GpuSampler sampler(static_cast<int>(rows.size()), 4096);
    const std::span<const TokenId> gpu = sampler.Sample(logits.Data(), kStride, kVocab, requests, stream.Get());

    for (std::size_t r = 0; r < rows.size(); ++r) {
        std::vector<float> cpu_logits = rows[r];
        const TokenId expected = Sampler(requests[r].params).Pick(cpu_logits, histories[r], requests[r].unit_draw);
        EXPECT_EQ(gpu[r], expected) << "row " << r;
    }
}

TEST(GpuSampler, PenalizesEachEarlierTokenOnce) {
    // Token 3 leads, but it was said before: with penalty 2 its 4.0 halves to
    // 2.0, below token 9's 3.0.
    std::vector<float> row(kVocab, -10.0f);
    row[3] = 4.0f;
    row[9] = 3.0f;
    DeviceBuffer<float> logits(row.size());
    logits.CopyFromHost(row);
    const std::vector<TokenId> said = {3};
    const SamplingRequest request{
        .params = {.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 2.0f},
        .penalty_ids = said};
    CudaStream stream;
    GpuSampler sampler(1, 16);
    EXPECT_EQ(sampler.Sample(logits.Data(), kVocab, kVocab, std::span(&request, 1), stream.Get())[0], 9);
}

TEST(GpuSampler, RejectsSettingsItCannotFollow) {
    DeviceBuffer<float> logits(kVocab);
    CudaStream stream;
    GpuSampler sampler(2, 16);
    const auto sample = [&](SamplingParams params, double draw = 0.5) {
        const SamplingRequest request{.params = params, .unit_draw = draw};
        return sampler.Sample(logits.Data(), kVocab, kVocab, std::span(&request, 1), stream.Get());
    };
    EXPECT_THROW(sample({.temperature = 0.7f, .top_k = 0}), std::invalid_argument);    // unlimited top-k
    EXPECT_THROW(sample({.temperature = 0.7f, .top_k = kMaxSamplingTopK + 1}), std::invalid_argument);
    EXPECT_THROW(sample({.temperature = 0.7f, .top_k = 20}, 1.0), std::invalid_argument);   // draw outside [0, 1)
    EXPECT_THROW(sample({.temperature = -1.0f, .top_k = 20}), std::invalid_argument);
    const std::vector<SamplingRequest> three(3);
    EXPECT_THROW(sampler.Sample(logits.Data(), kVocab, kVocab, three, stream.Get()), std::invalid_argument);
}

}  // namespace
}  // namespace murmur
