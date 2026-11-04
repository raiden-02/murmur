#include "engine/sampler.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

namespace murmur {
namespace {

SamplingParams Plain(float temperature = 1.0f) {
    return {.temperature = temperature, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 1.0f, .seed = 7};
}

std::vector<float> LogitsFor(const std::vector<double>& probabilities) {
    std::vector<float> logits;
    for (const double p : probabilities) {
        logits.push_back(static_cast<float>(std::log(p)));
    }
    return logits;
}

TEST(Sampler, ZeroTemperatureIsGreedy) {
    Sampler sampler(Plain(0.0f));
    std::vector<float> logits = {0.1f, 3.0f, 2.9f};
    EXPECT_EQ(sampler.Sample(logits, {}), 1);
}

TEST(Sampler, TopKOfOneIsGreedy) {
    SamplingParams params = Plain(5.0f);
    params.top_k = 1;
    Sampler sampler(params);
    for (int i = 0; i < 100; ++i) {
        std::vector<float> logits = {0.1f, 3.0f, 2.9f};
        EXPECT_EQ(sampler.Sample(logits, {}), 1);
    }
}

TEST(Sampler, SameSeedSameChoices) {
    Sampler a(Plain());
    Sampler b(Plain());
    for (int i = 0; i < 50; ++i) {
        std::vector<float> logits_a = LogitsFor({0.4, 0.3, 0.2, 0.1});
        std::vector<float> logits_b = logits_a;
        EXPECT_EQ(a.Sample(logits_a, {}), b.Sample(logits_b, {}));
    }
}

TEST(Sampler, DrawsInProportionToProbability) {
    Sampler sampler(Plain());
    const std::vector<double> probabilities = {0.5, 0.3, 0.2};
    std::array<int, 3> counts{};
    constexpr int kDraws = 20000;
    for (int i = 0; i < kDraws; ++i) {
        std::vector<float> logits = LogitsFor(probabilities);
        ++counts[sampler.Sample(logits, {})];
    }
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(counts[i]) / kDraws, probabilities[i], 0.02) << "token " << i;
    }
}

TEST(Sampler, TopPDropsTheUnlikelyTail) {
    // Probabilities 0.5, 0.3, 0.2 with top_p 0.6: the 0.2 tail fits inside the
    // 0.4 that may be dropped; 0.3 doesn't, so tokens 0 and 1 remain.
    SamplingParams params = Plain();
    params.top_p = 0.6f;
    Sampler sampler(params);
    std::array<int, 3> counts{};
    for (int i = 0; i < 5000; ++i) {
        std::vector<float> logits = LogitsFor({0.5, 0.3, 0.2});
        ++counts[sampler.Sample(logits, {})];
    }
    EXPECT_GT(counts[0], 0);
    EXPECT_GT(counts[1], 0);
    EXPECT_EQ(counts[2], 0);
}

TEST(Sampler, RepetitionPenaltyPushesSeenTokensDown) {
    SamplingParams params = Plain(0.0f);
    params.repetition_penalty = 1.1f;
    Sampler sampler(params);
    const std::vector<TokenId> history = {0, 0, 0};   // repeats count once
    std::vector<float> positive = {2.0f, 1.9f};        // 2.0 / 1.1 = 1.82 < 1.9
    EXPECT_EQ(sampler.Sample(positive, history), 1);
    std::vector<float> negative = {-1.0f, -1.05f};     // -1.0 * 1.1 = -1.1 < -1.05
    EXPECT_EQ(sampler.Sample(negative, history), 1);
}

TEST(Sampler, RejectsBadParameters) {
    EXPECT_THROW(Sampler({.temperature = -1.0f}), std::invalid_argument);
    EXPECT_THROW(Sampler({.top_k = -1}), std::invalid_argument);
    EXPECT_THROW(Sampler({.top_p = 0.0f}), std::invalid_argument);
    EXPECT_THROW(Sampler({.top_p = 1.5f}), std::invalid_argument);
    EXPECT_THROW(Sampler({.repetition_penalty = 0.0f}), std::invalid_argument);
    EXPECT_THROW(Sampler({.temperature = NAN}), std::invalid_argument);
}

}  // namespace
}  // namespace murmur
