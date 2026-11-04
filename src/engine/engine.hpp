#pragma once

#include "engine/chat_session.hpp"
#include "engine/gpu_sampler.hpp"
#include "engine/grammar.hpp"
#include "engine/kv_cache_manager.hpp"
#include "engine/sampler.hpp"
#include "engine/speculator.hpp"
#include "engine/transformer.hpp"
#include "model/chat_template.hpp"
#include "model/tokenizer.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace murmur {

using ConversationId = int;

struct ConversationOptions {
    std::string system_prompt;
    SamplingParams sampling;
    int priority = 0;            // higher is scheduled first and evicted last; Send can raise it per reply.
                                 // Above 0, its prompt may also use a whole step instead of a chunk
    int max_reply_tokens = 160;
    int memory_tokens = 0;       // the conversation's token budget (0: the whole context window); older
                                 // turns are forgotten to stay within it
    bool speculative = false;    // guess tokens with the engine's Speculator (if it has one) and check them
                                 // in one pass: faster replies, exactly the same text
    bool acts = false;           // end every reply with an action in the engine's ActionLanguage
};

// What NPCs can do, as a grammar the model is held to: after its words, the
// model writes `lead_in` and then one sentence of the grammar, which comes
// back as a kAction event. While it speaks, the model may not start a new
// line or open a parenthesis, so it can't write an action (or a stage
// direction) of its own into its words: the end of its turn is the only way
// out of them.
struct ActionLanguage {
    std::shared_ptr<const TokenGrammar> grammar;
    std::vector<TokenId> lead_in;   // e.g. "\nACTION:", written for the model before its action
};

enum class EngineEventKind {
    kText,            // the next piece of a reply
    kAction,          // the action that ends a reply (text: a sentence of the ActionLanguage), just before
                      // its kReplyFinished
    kReplyFinished,   // a reply is complete (or was cancelled); stats are filled
};

struct EngineEvent {
    ConversationId conversation;
    EngineEventKind kind;
    std::string text;
    ReplyStats stats{};
    bool drafted = false;   // kText: the draft model guessed this piece and the model confirmed it
    std::string forced;     // kAction: per byte of text, 1 if the grammar wrote it (jump-forward), else 0
};

struct EngineOptions {
    int max_prefill_tokens = 256;   // prompt tokens per step for ordinary conversations (chunked prefill);
                                    // a higher-priority prompt may use the whole step
    int draft_tokens = 4;           // tokens the Speculator guesses per step for a speculative conversation
    int max_speculative_batch = 4;  // speculate only while at most this many conversations decode: a bigger
                                    // batch already keeps the GPU busy, and guesses would only take its rows
    bool jump_forward = true;       // write the tokens an action's grammar leaves no choice about without
                                    // running the model for each (off: the model samples them, masked)
};

// What the engine is doing, for the X-ray view and the benchmarks.
struct EngineStats {
    static constexpr ConversationId kFree = -1;     // page owner: nobody
    static constexpr ConversationId kCached = -2;   // page owner: only the prefix cache

    std::vector<ConversationId> page_owner;   // per page: a conversation using it (any one, if shared)
    std::vector<std::uint8_t> page_shared;    // per page: 1 if more than one conversation uses it
    std::vector<ConversationId> last_batch;   // conversations in the last step, in planning order
    std::vector<int> last_batch_tokens;       // ...and how many tokens each ran (1: a decode token)
    int last_decode_rows = 0;
    int last_prefill_rows = 0;
    int last_draft_rows = 0;         // guessed tokens checked in the last step
    int last_accepted = 0;           // ...and confirmed
    double last_step_ms = 0.0;
    double last_draft_ms = 0.0;      // of which the draft model guessing
    std::int64_t steps = 0;
    std::int64_t generated_tokens = 0;
    std::int64_t evictions = 0;      // conversations whose KV cache was dropped to make room
    std::int64_t forgotten_turns = 0;  // old turns dropped to keep conversations within their budget
    std::int64_t reused_tokens = 0;  // prompt tokens served from the prefix cache instead of computed
    std::int64_t actions = 0;          // actions written (constrained decoding)
    std::int64_t action_tokens = 0;    // action tokens the model chose (under the grammar's mask)...
    std::int64_t forced_tokens = 0;    // ...and the ones the grammar decided, written without the model
    int draft_tokens = 0;              // guesses per step now (0: speculation off or no draft model)
    std::int64_t drafted_tokens = 0;   // tokens guessed by the draft model (speculative decoding)
    std::int64_t accepted_tokens = 0;  // ...and confirmed by the model
    int pages_free = 0;
    int pages_cached = 0;
    int page_tokens = 0;             // tokens one KV-cache page holds
    int max_batch = 0;               // conversations one step can decode
    int max_speculative_batch = 0;   // the draft model guesses only while at most this many decode
    WeightFormat weight_format = WeightFormat::kBf16;
    KvFormat kv_format = KvFormat::kBf16;
    std::size_t weight_bytes = 0;
    std::size_t kv_bytes = 0;
};

// Many conversations on one transformer, generated together with continuous
// batching: every step runs one forward pass over all the conversations
// that are decoding plus chunks of prompts waiting to be processed, so
// conversations join and leave the batch at any step. Not thread-safe: one
// thread drives it (InferenceService runs it on its worker).
class Engine {
public:
    Engine(Transformer& transformer, const Tokenizer& tokenizer, EngineOptions options = {},
           Speculator* speculator = nullptr);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void Open(ConversationId id, ConversationOptions options);
    void Close(ConversationId id);
    void Send(ConversationId id, std::string_view message, std::optional<int> priority = std::nullopt);
    void Cancel(ConversationId id);

    bool HasWork() const;
    bool IsReplying(ConversationId id) const;
    std::vector<EngineEvent> Step();
    EngineStats Stats() const;
    void Reconfigure(ModelWeights weights, std::size_t kv_bytes, KvFormat kv_format);
    void SetActionLanguage(ActionLanguage language);
    void SetDraftTokens(int draft_tokens);
    KvCacheManager& Cache() { return transformer_->KvCache(); }

private:
    using Clock = std::chrono::steady_clock;
    static constexpr int kActionBudget = 48;      // tokens a reply keeps free for its action (lead-in included)

    // Where a reply is: the model's words, then (for a conversation that
    // acts) the action under the grammar, then done.
    enum class ReplyPhase { kSpeaking, kActing, kActed };

    struct Conversation {
        ConversationId id;
        ConversationOptions options;
        Sampler sampler;
        std::vector<TokenId> tokens;   // the whole conversation; the first kv.length are cached
        std::size_t system_tokens = 0; // the system prompt's length: never forgotten
        std::vector<std::size_t> turn_starts;   // where each message from the player begins in `tokens`
        KvSequence kv;
        int published_pages = 0;       // full pages already offered to the prefix cache
        bool replying = false;
        bool cancelled = false;
        int reply_priority = 0;
        std::int64_t last_active = 0;  // step of the last forward pass it was part of

        // The distinct tokens so far, for the repetition penalty.
        std::vector<TokenId> distinct;
        std::unordered_set<TokenId> seen;

        // The reply in progress.
        std::unique_ptr<StreamingDecoder> decoder;
        ReplyStats stats;
        Clock::time_point sent_at;
        std::int64_t send_order = 0;   // replies in the order they were asked for: older ones go first
        Clock::time_point first_token_at;
        bool awaiting_first_token = false;
        int prompt_tokens = 0;
        // Random draws taken from the sampler ahead of use, to guess and check
        // several positions at once; consumed in order, so the draw sequence
        // is the same with or without speculation.
        std::vector<double> draws;
        std::vector<TokenId> step_tokens;   // this step's tokens when speculating: the newest, then the guesses
        // The action that ends the reply, once the model has finished speaking:
        // where it is in the grammar, where it began in `tokens`, and what it
        // wrote so far.
        ReplyPhase phase = ReplyPhase::kSpeaking;
        int grammar_state = 0;          // while kActing: a TokenGrammar state
        std::size_t action_start = 0;   // index in `tokens` of the lead-in
        std::string action;
        std::string action_forced;   // per byte of action: 1 if the grammar wrote it

        int Uncached() const { return static_cast<int>(tokens.size()) - kv.length; }
        int Priority() const { return replying ? reply_priority : options.priority; }
    };

    struct Planned {
        Conversation* conversation;
        int count;         // tokens to run this step
        int drafted = 0;   // of which guessed by the Speculator (they follow the newest token)
    };

    Conversation& Get(ConversationId id);
    void Append(Conversation& c, std::span<const TokenId> ids);
    void Rebuild(Conversation& c, std::size_t first_kept_turn);
    std::vector<Planned> Plan(std::vector<EngineEvent>& events);
    bool MakeRoom(Conversation& c, int length, const std::vector<Planned>& planned);
    void Evict(Conversation& c);
    void Speculate(std::vector<Planned>& plan);
    std::vector<StepEntry> Entries(const std::vector<Planned>& plan);
    std::span<const TokenId> SampleNext(const std::vector<Planned>& plan, const std::vector<StepEntry>& entries);
    void TakeTokens(const std::vector<Planned>& plan, const std::vector<StepEntry>& entries,
                    std::span<const TokenId> next, std::vector<EngineEvent>& events);
    void RecordBatch(const std::vector<Planned>& plan);
    void Verify(Conversation& c, int drafted, std::span<const TokenId> checked, std::vector<EngineEvent>& events);
    void Publish(Conversation& c);
    double TakeDraw(Conversation& c);
    std::span<const double> PeekDraws(Conversation& c, int count);
    void Accept(Conversation& c, TokenId token, std::vector<EngineEvent>& events, bool drafted = false);
    void StartAction(Conversation& c, std::vector<EngineEvent>& events);
    void AcceptAction(Conversation& c, TokenId token, std::vector<EngineEvent>& events);
    bool AppendActionToken(Conversation& c, TokenId token, int next_state, bool forced,
                           std::vector<EngineEvent>& events);
    void WriteForced(Conversation& c, std::vector<EngineEvent>& events);
    SamplingRequest SamplingFor(const Conversation& c, double draw, std::span<const TokenId> penalty_ids) const;
    void Finish(Conversation& c, bool truncated, std::vector<EngineEvent>& events);
    void Emit(Conversation& c, std::string text, std::vector<EngineEvent>& events, bool drafted = false);

    Transformer* transformer_;
    const Tokenizer* tokenizer_;
    EngineOptions options_;
    Speculator* speculator_;
    ChatTemplate template_;
    std::vector<TokenId> stop_tokens_;
    ActionLanguage actions_;
    DeviceBuffer<std::uint32_t> speech_mask_;   // while an acting conversation speaks: no line breaks or "("s
    GpuSampler sampler_;
    std::map<ConversationId, Conversation> conversations_;   // ordered: stable, deterministic scheduling
    std::int64_t sends_ = 0;
    EngineStats stats_;
};

}  // namespace murmur
