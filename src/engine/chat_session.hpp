#pragma once

#include "engine/sampler.hpp"
#include "engine/transformer.hpp"
#include "model/chat_template.hpp"
#include "model/tokenizer.hpp"

#include <functional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace murmur {

struct ReplyStats {
    int prompt_tokens = 0;              // tokens prefilled for this turn
    int reply_tokens = 0;               // tokens generated (each one is a decode step)
    double prefill_ms = 0.0;            // time to first token
    double decode_ms = 0.0;             // everything after the first token
    bool truncated = false;             // stopped by the length limit or a stop request
    bool context_reset = false;         // older turns were dropped to make room
    int action_tokens = 0;              // tokens of the action the model chose (constrained decoding)...
    int forced_tokens = 0;              // ...and the ones the grammar left no choice for, written without the model
    int drafted_tokens = 0;             // tokens the draft model guessed (speculative decoding)
    int accepted_tokens = 0;            // ...and how many of them the model confirmed
    double decode_bytes_per_token = 0;  // memory one decode step must read, mid-reply
    double peak_bandwidth = 0;          // the GPU's bytes per second

    double DecodeTokensPerSecond() const { return decode_ms > 0 ? reply_tokens / (decode_ms / 1e3) : 0.0; }
    double DecodeBandwidth() const { return DecodeTokensPerSecond() * decode_bytes_per_token; }
    double DecodeFractionOfPeak() const { return peak_bandwidth > 0 ? DecodeBandwidth() / peak_bandwidth : 0.0; }
};

// One character's conversation with the player, kept in the transformer's KV
// cache between turns so only the new message is processed each time.
// Uses the transformer exclusively: one session per transformer.
class ChatSession {
public:
    using TextCallback = std::function<void(std::string_view)>;

    ChatSession(Transformer& transformer, const Tokenizer& tokenizer, std::string system_prompt,
                SamplingParams sampling, int max_reply_tokens = 256);

    std::string Reply(std::string_view message, const TextCallback& on_text, std::stop_token stop = {},
                      ReplyStats* stats = nullptr);

    int TokensInContext() const { return static_cast<int>(context_.size()); }

private:
    TokenId NextToken();
    bool IsStopToken(TokenId id) const;
    void Reset();

    Transformer* transformer_;
    const Tokenizer* tokenizer_;
    ChatTemplate template_;
    std::string system_prompt_;
    Sampler sampler_;
    int max_reply_tokens_;
    std::vector<TokenId> stop_tokens_;

    std::vector<TokenId> context_;   // tokens whose keys/values are in the cache, in order
    std::vector<TokenId> pending_;   // tokens to prefill before the next message
    std::vector<float> logits_;      // host copy for sampling
    double peak_bandwidth_;
};

}  // namespace murmur
