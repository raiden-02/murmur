#pragma once

#include "engine/engine.hpp"
#include "engine/speculator.hpp"
#include "engine/transformer.hpp"

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace murmur {

inline constexpr ConversationId kNoConversation = -1;

enum class InferenceEventKind {
    kReady,           // the model finished loading
    kText,            // the next piece of a conversation's reply
    kAction,          // the action that ends a reply (text: a sentence of the action grammar)
    kReplyFinished,   // a reply is complete (or was cancelled)
    kReconfigured,    // the weights or the KV cache format changed (SetFormats)
    kError,           // a request failed and the engine carries on; text says why. For an engine-wide
                      // request (SetFormats, the optional draft model) conversation is kNoConversation
    kStopped,         // loading or a step failed and the worker stopped; text says why. Nothing follows
};

struct InferenceEvent {
    InferenceEventKind kind;
    ConversationId conversation = kNoConversation;   // kNoConversation for engine-wide events
    std::string text;
    bool drafted = false;   // kText: guessed by the draft model, confirmed by the model
    std::string forced;     // kAction: per byte of text, 1 if the grammar wrote it (jump-forward)
    ReplyStats stats{};     // filled for kReplyFinished
};

struct InferenceServiceOptions {
    std::filesystem::path model_dir;
    TransformerOptions transformer;
    EngineOptions engine;
    WeightFormat weights = WeightFormat::kBf16;
    std::size_t memory_budget = 0;   // weights + KV cache; 0: the KV cache is transformer.kv_cache_bytes
    // What conversations that act may do (ConversationOptions::acts): the
    // grammar's text (ByteAutomaton::Compile) and what the model is shown
    // before writing its action. Empty: no conversation acts.
    std::string action_grammar;
    std::string action_lead_in = "\nACTION:";
    // A small model of the same family that guesses tokens for speculative
    // conversations (ConversationOptions::speculative), with its own memory
    // outside the budget. Empty: no speculation.
    std::filesystem::path draft_model_dir;
    WeightFormat draft_weights = WeightFormat::kInt4;
};

// Runs the Engine on a background thread so the game loop never waits for
// the GPU. The game opens conversations, sends messages and drains results
// once per frame; none of the calls blocks on inference. All CUDA work
// (loading, generating, freeing) happens on the worker thread.
class InferenceService {
public:
    explicit InferenceService(InferenceServiceOptions options);
    ~InferenceService();

    InferenceService(const InferenceService&) = delete;
    InferenceService& operator=(const InferenceService&) = delete;

    ConversationId Open(ConversationOptions options);
    void Ask(ConversationId conversation, std::string message, std::optional<int> priority = std::nullopt);
    void Cancel(ConversationId conversation);
    void Close(ConversationId conversation);
    void SetFormats(WeightFormat weights, KvFormat kv);
    void SetDraftTokens(int draft_tokens);
    std::vector<InferenceEvent> TakeEvents();
    EngineStats Stats() const;

private:
    enum class CommandKind { kOpen, kAsk, kCancel, kClose, kFormats, kDraftTokens };
    struct Command {
        CommandKind kind;
        ConversationId conversation = kNoConversation;
        ConversationOptions options;
        std::string message;
        std::optional<int> priority;
        WeightFormat weights = WeightFormat::kBf16;
        KvFormat kv = KvFormat::kBf16;
        int draft_tokens = 0;
    };

    void Run(std::stop_token stop);
    void Apply(Engine& engine, Command& command);
    void Reconfigure(Engine& engine, const Command& command);
    void Enqueue(Command command);
    void Publish(InferenceEvent event);

    InferenceServiceOptions options_;
    std::optional<ModelConfig> config_;    // worker only, once loaded
    mutable std::mutex mutex_;
    std::condition_variable_any work_ready_;
    std::deque<Command> commands_;         // guarded by mutex_
    std::vector<InferenceEvent> events_;   // guarded by mutex_
    EngineStats stats_;                    // guarded by mutex_
    ConversationId next_id_ = 0;           // guarded by mutex_
    std::jthread worker_;                  // last: starts after, and stops before, everything above
};

}  // namespace murmur
