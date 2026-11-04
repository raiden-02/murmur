#pragma once

#include "engine/gpu_sampler.hpp"
#include "engine/sampler.hpp"
#include "engine/transformer.hpp"
#include "model/tokenizer.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace murmur {

// What the draft model needs to guess one conversation's next tokens.
struct DraftRequest {
    int conversation = 0;                    // which conversation (its draft cache is kept between calls)
    std::span<const TokenId> tokens;         // the conversation so far, its newest token included
    SamplingParams sampling;                 // the conversation's own settings...
    std::span<const double> draws;           // ...and the random draw of each position to guess: one token per draw
    std::span<const TokenId> penalty_ids;    // distinct tokens so far, for the repetition penalty
};

// A small model (same tokenizer, fewer and narrower layers) that guesses the
// next few tokens of conversations cheaply, so the main model can check all
// of them in one pass: speculative decoding. It keeps its own KV cache per
// conversation and, before guessing, catches up on whatever the
// conversation added since last time (or recomputes after the conversation
// forgot or was evicted). Guesses are sampled with the conversation's own
// settings and the same random draws the main model will use, so the two
// models agree as often as their probabilities allow.
class Speculator {
public:
    Speculator(std::unique_ptr<Transformer> draft, int candidate_count, std::span<const TokenId> stop_tokens);

    std::vector<std::vector<TokenId>> Propose(std::span<const DraftRequest> requests);
    void Forget(int conversation);
    const Transformer& Draft() const { return *draft_; }

private:
    struct Cached {
        KvSequence kv;
        std::vector<TokenId> tokens;   // what kv holds
        std::int64_t last_used = 0;
    };

    bool Prepare(Cached& cached, std::span<const TokenId> tokens, int length);
    void Run(std::span<Cached* const> sequences, std::span<const std::span<const TokenId>> tokens);

    std::unique_ptr<Transformer> draft_;
    int candidate_count_;
    std::vector<TokenId> stop_tokens_;
    GpuSampler sampler_;
    std::map<int, Cached> cached_;
    std::int64_t calls_ = 0;
};

}  // namespace murmur
