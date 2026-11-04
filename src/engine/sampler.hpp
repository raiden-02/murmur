#pragma once

#include "model/tokenizer.hpp"

#include <cstdint>
#include <random>
#include <span>

namespace murmur {

// How the next token is picked from the logits. The defaults are the
// model's own recommended settings (its generation_config.json).
struct SamplingParams {
    float temperature = 0.7f;          // 0 = greedy (always the top token)
    int top_k = 20;                    // keep the k most likely tokens; 0 = no limit
    float top_p = 0.8f;                // then the most likely ones covering this probability mass
    float repetition_penalty = 1.1f;   // > 1 discourages tokens already in the conversation
    std::uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(SamplingParams params);

    TokenId Sample(std::span<float> logits, std::span<const TokenId> history);
    TokenId Pick(std::span<float> logits, std::span<const TokenId> history, double unit_draw) const;
    double NextDraw();
    const SamplingParams& Params() const { return params_; }

private:
    SamplingParams params_;
    std::mt19937_64 rng_;
};

}  // namespace murmur
