#include "engine/chat_session.hpp"

#include "common/device_info.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace murmur {

namespace {

using Clock = std::chrono::steady_clock;

double MillisecondsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void Append(std::vector<TokenId>& out, const std::vector<TokenId>& ids) {
    out.insert(out.end(), ids.begin(), ids.end());
}

}  // namespace

ChatSession::ChatSession(Transformer& transformer, const Tokenizer& tokenizer, std::string system_prompt,
                         SamplingParams sampling, int max_reply_tokens)
    : transformer_(&transformer),
      tokenizer_(&tokenizer),
      template_(tokenizer),
      system_prompt_(std::move(system_prompt)),
      sampler_(sampling),
      max_reply_tokens_(max_reply_tokens),
      stop_tokens_{template_.EndOfTurn(), tokenizer.SpecialTokenId("<|endoftext|>")},
      logits_(tokenizer.VocabSize()),
      peak_bandwidth_(PeakMemoryBandwidth()) {
    if (max_reply_tokens <= 0 || max_reply_tokens >= transformer.Options().max_context) {
        throw std::invalid_argument("ChatSession: max_reply_tokens must be positive and fit the context window");
    }
    if (tokenizer.VocabSize() > transformer.Config().vocab_size) {
        throw std::invalid_argument("ChatSession: tokenizer and model vocabularies don't match");
    }
    Reset();
}

// Adds the player's message and generates the character's reply, calling
// `on_text` with each new piece of text as soon as it's complete UTF-8.
// Returns the whole reply.
//
// Only the new tokens are prefilled: the system prompt and earlier turns are
// already in the KV cache. If the conversation would outgrow the context
// window, older turns are dropped and it restarts from the system prompt.
// Generation ends at the model's end-of-turn token, after max_reply_tokens,
// or when `stop` is requested; the turn is closed properly in every case.
std::string ChatSession::Reply(std::string_view message, const TextCallback& on_text, std::stop_token stop,
                               ReplyStats* stats) {
    ReplyStats local;
    std::vector<TokenId> user_turn = template_.EncodeMessage({ChatRole::kUser, std::string(message)});
    Append(user_turn, template_.EncodeGenerationPrompt());

    const int capacity = transformer_->Options().max_context;
    const auto fits = [&] {
        return context_.size() + pending_.size() + user_turn.size() + max_reply_tokens_ <=
               static_cast<std::size_t>(capacity);
    };
    if (!fits()) {
        Reset();
        local.context_reset = true;
        if (!fits()) {
            throw std::length_error("ChatSession: message too long for the context window");
        }
    }

    try {
        std::vector<TokenId> prompt = std::exchange(pending_, {});
        Append(prompt, user_turn);
        const Clock::time_point start = Clock::now();
        transformer_->Forward(prompt, TokensInContext());
        Append(context_, prompt);
        local.prompt_tokens = static_cast<int>(prompt.size());

        std::string reply;
        StreamingDecoder decoder(*tokenizer_);
        const auto emit = [&](std::string text) {
            if (!text.empty()) {
                on_text(text);
                reply += text;
            }
        };
        bool ended = false;
        Clock::time_point first_token_time = start;
        for (int n = 0; n < max_reply_tokens_ && !stop.stop_requested(); ++n) {
            const TokenId next = NextToken();
            if (n == 0) {
                local.prefill_ms = MillisecondsSince(start);
                first_token_time = Clock::now();
            }
            if (IsStopToken(next)) {
                ended = true;
                break;
            }
            emit(decoder.Push(next));
            ++local.reply_tokens;
            transformer_->Forward(std::span(&next, 1), TokensInContext());
            context_.push_back(next);
        }
        emit(decoder.Flush());
        local.decode_ms = MillisecondsSince(first_token_time);
        local.truncated = !ended;
        local.decode_bytes_per_token =
            static_cast<double>(transformer_->DecodeBytes(TokensInContext() - local.reply_tokens / 2));
        local.peak_bandwidth = peak_bandwidth_;

        // Close the turn: the end-of-turn marker (the model's own, or ours when
        // it was cut off) and the newline after it go in with the next message.
        pending_.push_back(template_.EndOfTurn());
        Append(pending_, template_.EncodeAfterReply());
        if (stats != nullptr) {
            *stats = local;
        }
        return reply;
    } catch (...) {
        Reset();   // the cache no longer matches context_; start clean next time
        throw;
    }
}

TokenId ChatSession::NextToken() {
    const SamplingParams& p = sampler_.Params();
    if (p.temperature == 0.0f && p.repetition_penalty == 1.0f) {
        return transformer_->ArgmaxLastLogits(tokenizer_->VocabSize());   // pure greedy stays on the GPU
    }
    transformer_->CopyLastLogits(logits_);
    return sampler_.Sample(logits_, context_);
}

bool ChatSession::IsStopToken(TokenId id) const {
    return std::ranges::find(stop_tokens_, id) != stop_tokens_.end();
}

// Forgets the conversation: the next message starts after a fresh system prompt.
void ChatSession::Reset() {
    context_.clear();
    pending_ = template_.EncodeMessage({ChatRole::kSystem, system_prompt_});
}

}  // namespace murmur
