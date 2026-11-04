#include "engine/inference_service.hpp"

#include "engine/grammar.hpp"
#include "engine/model_loader.hpp"
#include "model/chat_template.hpp"

#include <exception>
#include <memory>
#include <optional>
#include <utility>

namespace murmur {
namespace {

InferenceEventKind ServiceKind(EngineEventKind kind) {
    switch (kind) {
    case EngineEventKind::kText:
        return InferenceEventKind::kText;
    case EngineEventKind::kAction:
        return InferenceEventKind::kAction;
    case EngineEventKind::kReplyFinished:
        break;
    }
    return InferenceEventKind::kReplyFinished;
}

}  // namespace

// Starts the worker thread, which loads the model in the background and
// publishes kReady (or kError) when it's done.
InferenceService::InferenceService(InferenceServiceOptions options)
    : options_(std::move(options)), worker_([this](std::stop_token stop) { Run(stop); }) {}

// std::jthread requests a stop and joins. Replies in progress stop after the
// current step; a model still loading finishes loading first (about 2 s).
InferenceService::~InferenceService() {
    worker_.request_stop();
}

// Starts a conversation and returns its id at once; requests for it may
// follow immediately (they are applied in order on the worker).
ConversationId InferenceService::Open(ConversationOptions options) {
    ConversationId id;
    {
        const std::lock_guard lock(mutex_);
        id = next_id_++;
    }
    Enqueue({.kind = CommandKind::kOpen, .conversation = id, .options = std::move(options)});
    return id;
}

// Sends a message; the reply arrives as kText events followed by one
// kReplyFinished (or kError then kReplyFinished if it couldn't start).
// `priority` overrides the conversation's for this reply.
void InferenceService::Ask(ConversationId conversation, std::string message, std::optional<int> priority) {
    Enqueue({.kind = CommandKind::kAsk, .conversation = conversation, .message = std::move(message),
             .priority = priority});
}

// Stops the conversation's reply in progress (it still ends with kReplyFinished).
void InferenceService::Cancel(ConversationId conversation) {
    Enqueue({.kind = CommandKind::kCancel, .conversation = conversation});
}

// Forgets the conversation and frees its memory; no further events for it.
void InferenceService::Close(ConversationId conversation) {
    Enqueue({.kind = CommandKind::kClose, .conversation = conversation});
}

// Switches the model to `weights` and its KV cache to `kv`, between steps;
// kReconfigured follows when done (reloading takes a second or two, while
// no step runs). With a memory budget the KV cache grows or shrinks to fill
// what the weights leave. Conversations keep going: their cached keys and
// values are recomputed, as after an eviction.
void InferenceService::SetFormats(WeightFormat weights, KvFormat kv) {
    Enqueue({.kind = CommandKind::kFormats, .weights = weights, .kv = kv});
}

// How many tokens the draft model guesses per step for speculative
// conversations (0 turns speculation off), from the next step on.
void InferenceService::SetDraftTokens(int draft_tokens) {
    Enqueue({.kind = CommandKind::kDraftTokens, .draft_tokens = draft_tokens});
}

// Everything published since the last call, oldest first.
std::vector<InferenceEvent> InferenceService::TakeEvents() {
    const std::lock_guard lock(mutex_);
    return std::exchange(events_, {});
}

// The engine's state after its latest step.
EngineStats InferenceService::Stats() const {
    const std::lock_guard lock(mutex_);
    return stats_;
}

void InferenceService::Enqueue(Command command) {
    {
        const std::lock_guard lock(mutex_);
        commands_.push_back(std::move(command));
    }
    work_ready_.notify_one();
}

// The worker: load once, then alternate between applying the game's requests
// and running engine steps until the service stops. A failed load or step
// ends the worker (kStopped); a failed request is reported (kError) and the
// worker carries on, and so does a draft model that fails to load: it is
// optional, so the engine runs without guesses.
void InferenceService::Run(std::stop_token stop) {
    std::optional<LoadedModel> model;
    std::unique_ptr<Speculator> speculator;
    std::optional<Engine> engine;
    try {
        model.emplace(LoadModel(options_.model_dir, options_.transformer, options_.weights, options_.memory_budget));
        config_ = model->transformer->Config();
        if (!options_.draft_model_dir.empty()) {
            try {
                speculator = LoadSpeculator(options_.draft_model_dir, options_.draft_weights, model->tokenizer);
            } catch (const std::exception& error) {
                Publish({.kind = InferenceEventKind::kError,
                         .text = std::string("draft model not loaded, no guesses: ") + error.what()});
            }
        }
        model->transformer->RecordDecodeGraphs();
        engine.emplace(*model->transformer, model->tokenizer, options_.engine, speculator.get());
        if (!options_.action_grammar.empty()) {
            const ChatTemplate chat(model->tokenizer);
            engine->SetActionLanguage(
                {.grammar = std::make_shared<TokenGrammar>(ByteAutomaton::Compile(options_.action_grammar),
                                                           model->tokenizer, chat.EndOfTurn()),
                 .lead_in = model->tokenizer.Encode(options_.action_lead_in, SpecialTokens::kAsText)});
        }
    } catch (const std::exception& error) {
        Publish({.kind = InferenceEventKind::kStopped, .text = error.what()});
        return;
    }
    {
        EngineStats stats = engine->Stats();   // the empty cache and the formats, before any step
        const std::lock_guard lock(mutex_);
        stats_ = std::move(stats);
    }
    Publish({.kind = InferenceEventKind::kReady});

    while (!stop.stop_requested()) {
        std::deque<Command> commands;
        {
            std::unique_lock lock(mutex_);
            const bool busy = engine->HasWork();
            if (!work_ready_.wait(lock, stop, [&] { return busy || !commands_.empty(); })) {
                return;   // the service is shutting down
            }
            commands.swap(commands_);
        }
        for (Command& command : commands) {
            Apply(*engine, command);
        }
        if (!model->transformer->HasKvCache()) {   // a format switch couldn't even restore the old cache
            Publish({.kind = InferenceEventKind::kStopped, .text = "out of GPU memory for the KV cache"});
            return;
        }
        if (!engine->HasWork()) {
            continue;
        }
        try {
            std::vector<EngineEvent> events = engine->Step();
            EngineStats stats = engine->Stats();
            const std::lock_guard lock(mutex_);
            for (EngineEvent& event : events) {
                events_.push_back({.kind = ServiceKind(event.kind),
                                   .conversation = event.conversation,
                                   .text = std::move(event.text),
                                   .drafted = event.drafted,
                                   .forced = std::move(event.forced),
                                   .stats = event.stats});
            }
            stats_ = std::move(stats);
        } catch (const std::exception& error) {
            Publish({.kind = InferenceEventKind::kStopped, .text = error.what()});
            return;
        }
    }
}

void InferenceService::Apply(Engine& engine, Command& command) {
    try {
        switch (command.kind) {
        case CommandKind::kOpen:
            engine.Open(command.conversation, std::move(command.options));
            break;
        case CommandKind::kAsk:
            engine.Send(command.conversation, command.message, command.priority);
            break;
        case CommandKind::kCancel:
            engine.Cancel(command.conversation);
            break;
        case CommandKind::kClose:
            engine.Close(command.conversation);
            break;
        case CommandKind::kFormats:
            Reconfigure(engine, command);
            break;
        case CommandKind::kDraftTokens:
            engine.SetDraftTokens(command.draft_tokens);
            break;
        }
    } catch (const std::exception& error) {
        Publish({.kind = InferenceEventKind::kError, .conversation = command.conversation, .text = error.what()});
        // Every Ask ends with one kReplyFinished: here, unless the rejected
        // Ask found an earlier reply still running, whose own end follows.
        if (command.kind == CommandKind::kAsk && !engine.IsReplying(command.conversation)) {
            Publish({.kind = InferenceEventKind::kReplyFinished, .conversation = command.conversation});
        }
    }
}

void InferenceService::Reconfigure(Engine& engine, const Command& command) {
    const EngineStats current = engine.Stats();
    ModelWeights weights = LoadWeights(options_.model_dir, *config_, command.weights);
    const std::size_t kv_bytes =
        options_.memory_budget > 0 ? KvBytesWithin(options_.memory_budget, weights) : current.kv_bytes;
    engine.Reconfigure(std::move(weights), kv_bytes, command.kv);
    EngineStats stats = engine.Stats();
    const std::lock_guard lock(mutex_);
    stats_ = std::move(stats);
    events_.push_back({.kind = InferenceEventKind::kReconfigured});
}

void InferenceService::Publish(InferenceEvent event) {
    const std::lock_guard lock(mutex_);
    events_.push_back(std::move(event));
}

}  // namespace murmur
