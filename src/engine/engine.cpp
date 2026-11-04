#include "engine/engine.hpp"

#include "common/device_info.hpp"

#include <algorithm>
#include <deque>
#include <stdexcept>
#include <utility>

namespace murmur {

namespace {

double MillisecondsBetween(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

// `speculator` (optional, owned by the caller) guesses tokens for
// conversations that ask for speculative decoding.
Engine::Engine(Transformer& transformer, const Tokenizer& tokenizer, EngineOptions options, Speculator* speculator)
    : transformer_(&transformer),
      tokenizer_(&tokenizer),
      options_(options),
      speculator_(speculator),
      template_(tokenizer),
      stop_tokens_{template_.EndOfTurn(), tokenizer.SpecialTokenId("<|endoftext|>")},
      sampler_(transformer.Options().max_logit_rows,
               transformer.Options().max_logit_rows * transformer.Options().max_context) {
    if (options.max_prefill_tokens <= 0 || options.draft_tokens < 0 || options.max_speculative_batch < 0) {
        throw std::invalid_argument(
            "Engine: max_prefill_tokens must be positive, draft_tokens and max_speculative_batch not negative");
    }
    if (tokenizer.VocabSize() > transformer.Config().vocab_size) {
        throw std::invalid_argument("Engine: tokenizer and model vocabularies don't match");
    }
}

// Closes every conversation still open: the KV cache belongs to the
// transformer and outlives the engine, so their pages must go back to it
// (what they published stays in the prefix cache for the next engine).
Engine::~Engine() {
    while (!conversations_.empty()) {
        Close(conversations_.begin()->first);
    }
}

// Starts a conversation whose first turn is the system prompt. Nothing runs
// until the first Send; the system prompt is prefilled with it (or reused
// from the prefix cache when another conversation already computed it).
void Engine::Open(ConversationId id, ConversationOptions options) {
    if (conversations_.contains(id)) {
        throw std::invalid_argument("Engine::Open: conversation id already in use");
    }
    const int window = transformer_->Options().max_context;
    if (options.memory_tokens <= 0 || options.memory_tokens > window) {
        options.memory_tokens = window;
    }
    if (options.max_reply_tokens <= 0 || options.max_reply_tokens >= options.memory_tokens) {
        throw std::invalid_argument("Engine::Open: max_reply_tokens must be positive and fit the memory budget");
    }
    const SamplingParams& p = options.sampling;
    if (p.temperature > 0.0f && (p.top_k < 1 || p.top_k > kMaxSamplingTopK)) {
        throw std::invalid_argument("Engine::Open: sampling needs top_k between 1 and " +
                                    std::to_string(kMaxSamplingTopK));
    }
    Conversation c{.id = id, .options = options, .sampler = Sampler(options.sampling)};
    Append(c, template_.EncodeMessage({ChatRole::kSystem, c.options.system_prompt}));
    c.system_tokens = c.tokens.size();
    const std::size_t reply_room = static_cast<std::size_t>(options.max_reply_tokens) + (options.acts ? kActionBudget : 0);
    if (c.system_tokens + reply_room >= static_cast<std::size_t>(options.memory_tokens)) {
        throw std::invalid_argument("Engine::Open: the system prompt leaves no room for a reply in the memory budget");
    }
    conversations_.emplace(id, std::move(c));
}

// Forgets a conversation and frees its pages (a reply in progress just stops).
void Engine::Close(ConversationId id) {
    Conversation& c = Get(id);
    Cache().Release(c.kv);
    if (speculator_ != nullptr) {
        speculator_->Forget(id);
    }
    conversations_.erase(id);
}

// Adds a message from the player and starts the character's reply, which
// arrives as events from the following steps; `priority` overrides the
// conversation's for this reply. If the conversation would outgrow its
// memory budget, its oldest turns are forgotten: it keeps the system prompt
// and the latest turns that fit in half of the rest of the budget (less,
// if the new turn and its reply need more than the other half), and
// recomputes them (the system prompt usually comes straight from the prefix
// cache).
void Engine::Send(ConversationId id, std::string_view message, std::optional<int> priority) {
    Conversation& c = Get(id);
    if (c.replying) {
        throw std::logic_error("Engine::Send: the previous reply hasn't finished");
    }
    if (c.options.acts && !actions_.grammar) {
        throw std::logic_error("Engine::Send: the conversation acts, but the engine has no action language");
    }
    std::vector<TokenId> turn = template_.EncodeMessage({ChatRole::kUser, std::string(message)});
    const std::vector<TokenId> prompt = template_.EncodeGenerationPrompt();
    turn.insert(turn.end(), prompt.begin(), prompt.end());

    c.stats = {};
    const std::size_t budget = static_cast<std::size_t>(c.options.memory_tokens);
    const std::size_t needed = turn.size() + c.options.max_reply_tokens + (c.options.acts ? kActionBudget : 0);
    if (c.system_tokens + needed > budget) {
        throw std::length_error("Engine::Send: message too long for the conversation's memory");
    }
    if (c.tokens.size() + needed > budget) {
        // Keep the newest turns that fit in half of what the system prompt
        // leaves (so the next message doesn't have to forget again at once),
        // and never so many that the new turn and its reply don't fit.
        const std::size_t room = budget - c.system_tokens;
        const std::size_t keep = std::min(room / 2, room - needed);
        std::size_t first_kept = c.turn_starts.size();
        while (first_kept > 0 && c.tokens.size() - c.turn_starts[first_kept - 1] <= keep) {
            --first_kept;
        }
        Rebuild(c, first_kept);
        c.stats.context_reset = true;
    }
    c.turn_starts.push_back(c.tokens.size());
    Append(c, turn);
    c.reply_priority = priority.value_or(c.options.priority);
    c.replying = true;
    c.cancelled = false;
    c.sent_at = Clock::now();
    c.send_order = sends_++;
    c.awaiting_first_token = true;
    c.prompt_tokens = c.Uncached();
    c.decoder = std::make_unique<StreamingDecoder>(*tokenizer_);
    c.phase = ReplyPhase::kSpeaking;
    c.action.clear();
    c.action_forced.clear();
}

// Stops a reply at the next step; it still ends with kReplyFinished.
void Engine::Cancel(ConversationId id) {
    Conversation& c = Get(id);
    c.cancelled = c.replying;
}

// Whether a reply of conversation `id` is still running (false for an
// unknown id).
bool Engine::IsReplying(ConversationId id) const {
    const auto it = conversations_.find(id);
    return it != conversations_.end() && it->second.replying;
}

bool Engine::HasWork() const {
    return std::ranges::any_of(conversations_, [](const auto& item) { return item.second.replying; });
}

// One step of continuous batching: plans which conversations run and how
// many tokens each, lets the Speculator guess ahead for speculative
// conversations, runs them all in one forward pass, samples the next token
// of every conversation that reached the end of its prompt (and checks the
// guesses), and returns what happened (text pieces and finished replies).
std::vector<EngineEvent> Engine::Step() {
    std::vector<EngineEvent> events;
    std::vector<Planned> plan = Plan(events);
    if (plan.empty()) {
        RecordBatch(plan);
        return events;
    }
    const Clock::time_point start = Clock::now();
    Speculate(plan);
    stats_.last_draft_ms = MillisecondsBetween(start, Clock::now());
    const std::vector<StepEntry> entries = Entries(plan);
    transformer_->Step(entries);
    const std::span<const TokenId> next = SampleNext(plan, entries);
    TakeTokens(plan, entries, next, events);
    stats_.last_step_ms = MillisecondsBetween(start, Clock::now());
    RecordBatch(plan);
    ++stats_.steps;
    return events;
}

// The forward pass's input: each planned conversation's next tokens (its
// newest and the draft's guesses, when speculating), and whether it wants
// logits: only an entry that reaches the end of its uncached tokens does.
std::vector<StepEntry> Engine::Entries(const std::vector<Planned>& plan) {
    std::vector<StepEntry> entries;
    entries.reserve(plan.size());
    for (const Planned& p : plan) {
        Conversation& c = *p.conversation;
        if (p.drafted > 0) {
            entries.push_back(
                {.sequence = &c.kv, .tokens = c.step_tokens, .logits = p.count, .decode_each_token = true});
        } else {
            entries.push_back({.sequence = &c.kv,
                               .tokens = std::span(c.tokens).subspan(c.kv.length, p.count),
                               .logits = p.count == c.Uncached() ? 1 : 0});
        }
    }
    return entries;
}

// After the forward pass: advances what is cached, then samples one token
// per logits row, all on the GPU in one launch. A speculative conversation's
// row r predicts the token after its r-th step token, with that position's
// own draw and the guesses before it counted for the repetition penalty:
// exactly what decoding one token at a time would have used. (Its cache
// length is settled by Verify, once it's known which guesses were kept.)
std::span<const TokenId> Engine::SampleNext(const std::vector<Planned>& plan, const std::vector<StepEntry>& entries) {
    std::vector<SamplingRequest> requests;
    std::deque<std::vector<TokenId>> penalties;   // stable addresses for the requests' spans
    for (std::size_t i = 0; i < plan.size(); ++i) {
        Conversation& c = *plan[i].conversation;
        c.last_active = stats_.steps;
        if (plan[i].drafted > 0) {
            const std::span<const double> draws = PeekDraws(c, plan[i].count);
            std::vector<TokenId> penalty = c.distinct;
            for (int row = 0; row < plan[i].count; ++row) {
                if (row > 0 && std::ranges::find(penalty, c.step_tokens[row]) == penalty.end()) {
                    penalty.push_back(c.step_tokens[row]);
                }
                penalties.push_back(penalty);
                requests.push_back(SamplingFor(c, draws[row], penalties.back()));
            }
            continue;
        }
        // The forward pass wrote the tokens' keys and values: they are cached now.
        c.kv.length += plan[i].count;
        Publish(c);
        if (entries[i].logits > 0) {
            requests.push_back(SamplingFor(c, TakeDraw(c), c.distinct));
        }
    }
    if (requests.empty()) {
        return {};
    }
    return sampler_.Sample(transformer_->StepLogits(), transformer_->Config().vocab_size, tokenizer_->VocabSize(),
                           requests, transformer_->Stream());
}

// Hands each conversation its sampled token (or, when speculating, checks
// the draft's guesses against them).
void Engine::TakeTokens(const std::vector<Planned>& plan, const std::vector<StepEntry>& entries,
                        std::span<const TokenId> next, std::vector<EngineEvent>& events) {
    stats_.last_draft_rows = 0;
    stats_.last_accepted = 0;
    std::size_t row = 0;
    for (std::size_t i = 0; i < plan.size(); ++i) {
        Conversation& c = *plan[i].conversation;
        if (plan[i].drafted > 0) {
            Verify(c, plan[i].drafted, next.subspan(row, plan[i].count), events);
            row += plan[i].count;
        } else if (entries[i].logits > 0) {
            Accept(c, next[row++], events);
        }
    }
}

// The step's batch, for the X-ray and the benchmarks.
void Engine::RecordBatch(const std::vector<Planned>& plan) {
    stats_.last_batch.clear();
    stats_.last_batch_tokens.clear();
    stats_.last_decode_rows = 0;
    stats_.last_prefill_rows = 0;
    for (const Planned& p : plan) {
        stats_.last_batch.push_back(p.conversation->id);
        stats_.last_batch_tokens.push_back(p.count);
        if (p.drafted > 0) {
            ++stats_.last_decode_rows;
            stats_.last_draft_rows += p.drafted;
        } else {
            (p.count == 1 ? stats_.last_decode_rows : stats_.last_prefill_rows) += p.count;
        }
    }
}

// Lets the Speculator guess the next tokens of every planned speculative
// conversation that is decoding, all at once, within what the step has room
// for (rows, logits rows, the reply's length limit). A conversation's
// guesses follow its newest token in the step; pages for them come only
// from free memory, never by evicting anyone, and without room it decodes
// one token as usual. Nobody speculates while more than
// max_speculative_batch conversations decode.
void Engine::Speculate(std::vector<Planned>& plan) {
    if (speculator_ == nullptr || options_.draft_tokens == 0) {
        return;
    }
    const TransformerOptions& limits = transformer_->Options();
    int rows = 0;
    int logit_rows = 0;
    int decoding = 0;
    for (const Planned& p : plan) {
        rows += p.count;
        logit_rows += p.count == p.conversation->Uncached() ? 1 : 0;
        decoding += p.count == 1 && p.conversation->Uncached() == 1 ? 1 : 0;
    }
    const std::size_t max_requests = static_cast<std::size_t>(
        std::min(options_.max_speculative_batch, speculator_->Draft().Options().max_batch));
    if (decoding > options_.max_speculative_batch) {
        return;
    }
    std::vector<DraftRequest> requests;
    std::vector<std::size_t> planned;
    for (std::size_t i = 0; i < plan.size() && requests.size() < max_requests; ++i) {
        Conversation& c = *plan[i].conversation;
        if (!c.options.speculative || plan[i].count != 1 || c.Uncached() != 1 || c.phase != ReplyPhase::kSpeaking) {
            continue;
        }
        const int count = std::min({options_.draft_tokens, c.options.max_reply_tokens - c.stats.reply_tokens - 1,
                                    limits.max_chunk - rows, limits.max_logit_rows - logit_rows,
                                    limits.max_context - c.kv.length - 1});
        if (count <= 0) {
            continue;
        }
        requests.push_back({.conversation = c.id,
                            .tokens = c.tokens,
                            .sampling = c.options.sampling,
                            .draws = PeekDraws(c, count),
                            .penalty_ids = c.distinct});
        planned.push_back(i);
        rows += count;
        logit_rows += count;
    }
    if (requests.empty()) {
        return;
    }
    const std::vector<std::vector<TokenId>> guesses = speculator_->Propose(requests);
    for (std::size_t r = 0; r < guesses.size(); ++r) {
        Planned& p = plan[planned[r]];
        Conversation& c = *p.conversation;
        const int drafted = static_cast<int>(guesses[r].size());
        if (drafted == 0 || !Cache().Reserve(c.kv, c.kv.length + 1 + drafted)) {
            continue;
        }
        c.step_tokens.assign(1, c.tokens.back());
        c.step_tokens.insert(c.step_tokens.end(), guesses[r].begin(), guesses[r].end());
        p.count = 1 + drafted;
        p.drafted = drafted;
    }
}

// Checks a speculative conversation's guesses against what the model
// sampled at each position (`checked`, one per step token). Guesses are
// accepted while they equal the model's own choice; the model's token at the
// first mismatch (or after the last guess) comes free with the same pass.
// Every accepted token goes through Accept as if decoded alone, so the text
// is exactly what one-at-a-time decoding would produce. Cached positions
// past the accepted ones are dropped.
void Engine::Verify(Conversation& c, int drafted, std::span<const TokenId> checked, std::vector<EngineEvent>& events) {
    const int before = c.kv.length;
    c.kv.length = before + 1 + drafted;   // the pass cached the newest token and every guess
    int accepted = 0;
    while (accepted < drafted && checked[accepted] == c.step_tokens[1 + accepted]) {
        ++accepted;
    }
    c.stats.drafted_tokens += drafted;
    c.stats.accepted_tokens += accepted;
    stats_.drafted_tokens += drafted;
    stats_.accepted_tokens += accepted;
    stats_.last_accepted += accepted;
    // Speculation only runs while the model speaks; once it ends its words
    // (and starts an action) the rest of the guesses don't count.
    for (int j = 0; j <= accepted && c.replying && c.phase == ReplyPhase::kSpeaking; ++j) {
        static_cast<void>(TakeDraw(c));   // this position's draw is used, as decoding alone would have
        Accept(c, checked[j], events, j < accepted);
    }
    // Keep the cached positions that still match the conversation's tokens.
    int valid = 0;
    while (valid <= drafted && static_cast<std::size_t>(before + valid) < c.tokens.size() &&
           c.tokens[before + valid] == c.step_tokens[valid]) {
        ++valid;
    }
    c.kv.length = before + valid;
    Cache().Truncate(c.kv, c.kv.length);
    Publish(c);
}

// Full pages that just filled up can be shared with later prompts.
void Engine::Publish(Conversation& c) {
    const int full_pages = c.kv.length / Cache().PageSize();
    if (full_pages > c.published_pages) {
        Cache().Publish(c.kv, std::span(c.tokens).first(static_cast<std::size_t>(full_pages) * Cache().PageSize()));
        c.published_pages = full_pages;
    }
}

// The random draw for the conversation's next sampled position: one drawn
// ahead for speculation if there is one, else a new one (0 when greedy).
double Engine::TakeDraw(Conversation& c) {
    if (!c.draws.empty()) {
        const double draw = c.draws.front();
        c.draws.erase(c.draws.begin());
        return draw;
    }
    return c.options.sampling.temperature > 0.0f ? c.sampler.NextDraw() : 0.0;
}

// The draws of the conversation's next `count` positions, drawn ahead as
// needed; they stay queued until those positions are sampled.
std::span<const double> Engine::PeekDraws(Conversation& c, int count) {
    while (static_cast<int>(c.draws.size()) < count) {
        c.draws.push_back(c.options.sampling.temperature > 0.0f ? c.sampler.NextDraw() : 0.0);
    }
    return std::span(c.draws).first(static_cast<std::size_t>(count));
}

// Chooses this step's work. Cancelled replies end first. Then every
// conversation with tokens to run is considered in order: higher priority
// first; within a priority, decoding conversations (one token, keeps their
// text flowing) before prompts, which go in arrival order. Prompts of
// ordinary priority share a budget of max_prefill_tokens per step and are cut
// into chunks to fit (chunked prefill), so a long prompt never stalls
// everyone else's replies. A prompt of higher priority (the player's) may use
// the whole step: its first word matters more than one step of chatter.
std::vector<Engine::Planned> Engine::Plan(std::vector<EngineEvent>& events) {
    std::vector<Conversation*> runnable;
    for (auto& [id, c] : conversations_) {
        if (c.replying && c.cancelled) {
            Finish(c, true, events);
        } else if (c.replying && c.Uncached() > 0) {
            runnable.push_back(&c);
        }
    }
    std::ranges::sort(runnable, [](const Conversation* a, const Conversation* b) {
        if (a->Priority() != b->Priority()) {
            return a->Priority() > b->Priority();
        }
        const bool a_decodes = a->Uncached() == 1;
        const bool b_decodes = b->Uncached() == 1;
        if (a_decodes != b_decodes) {
            return a_decodes;
        }
        return a_decodes ? a->last_active < b->last_active : a->sent_at < b->sent_at;
    });

    const TransformerOptions& limits = transformer_->Options();
    int rows_left = limits.max_chunk;
    int prefill_left = options_.max_prefill_tokens;
    std::vector<Planned> plan;
    for (Conversation* c : runnable) {
        if (static_cast<int>(plan.size()) == limits.max_batch || rows_left == 0) {
            break;
        }
        if (c->kv.length == 0 && c->kv.pages.empty()) {
            // Starting (or restarting after eviction): reuse whatever prefix
            // of this conversation is still in the prefix cache.
            const int reused = Cache().AttachPrefix(c->kv, c->tokens);
            c->published_pages = reused / Cache().PageSize();
            stats_.reused_tokens += reused;
        }
        const int uncached = c->Uncached();
        const int budget = c->Priority() > 0 ? rows_left : std::min(prefill_left, rows_left);
        const int count = uncached == 1 ? 1 : std::min(uncached, budget);
        if (c->kv.length + count > limits.max_context) {
            // Send's budget check rules this out; should it ever happen, one
            // reply ends early rather than the step failing for everyone.
            Finish(*c, true, events);
            continue;
        }
        if (count <= 0 || !MakeRoom(*c, c->kv.length + count, plan)) {
            continue;
        }
        plan.push_back({c, count});
        rows_left -= count;
        if (uncached > 1) {
            prefill_left = std::max(0, prefill_left - count);   // a priority prompt may overdraw it
        }
    }
    return plan;
}

// Gives `c` pages for `length` positions. When the cache is full (after
// reclaiming unused cached pages), other conversations' KV caches are
// dropped: lowest priority first, idle before replying, then idle ones
// least recently active first and replying ones newest first; never one
// already planned for this step. A conversation may drop idle ones of its
// own priority or lower, replying ones of lower priority, and replying ones
// of its own priority only if they were asked after it. So the oldest reply
// can always make progress (when every reply holds half a prompt and the
// cache is full, someone must give way), and two replies never take turns
// evicting each other.
// Dropped conversations keep their tokens and recompute them when they next
// run (their published pages may still be cached by then).
bool Engine::MakeRoom(Conversation& c, int length, const std::vector<Planned>& planned) {
    if (Cache().Reserve(c.kv, length)) {
        return true;
    }
    std::vector<Conversation*> victims;
    for (auto& [id, other] : conversations_) {
        const bool in_step = std::ranges::any_of(planned, [&](const Planned& p) { return p.conversation == &other; });
        if (&other == &c || other.kv.pages.empty() || in_step || other.Priority() > c.Priority()) {
            continue;
        }
        if (!other.replying || other.Priority() < c.Priority() || other.send_order > c.send_order) {
            victims.push_back(&other);
        }
    }
    std::ranges::sort(victims, [](const Conversation* a, const Conversation* b) {
        if (a->Priority() != b->Priority()) {
            return a->Priority() < b->Priority();
        }
        if (a->replying != b->replying) {
            return !a->replying;
        }
        return a->replying ? a->send_order > b->send_order : a->last_active < b->last_active;
    });
    for (Conversation* victim : victims) {
        Evict(*victim);
        if (Cache().Reserve(c.kv, length)) {
            return true;
        }
    }
    return false;
}

void Engine::Evict(Conversation& c) {
    Cache().Release(c.kv);
    c.published_pages = 0;
    ++stats_.evictions;
}

// Takes the token sampled for `c`: either the end of the reply, or the next
// piece of it (which runs through the model in the next step).
void Engine::Accept(Conversation& c, TokenId token, std::vector<EngineEvent>& events, bool drafted) {
    if (c.awaiting_first_token) {
        c.awaiting_first_token = false;
        c.first_token_at = Clock::now();
        c.stats.prefill_ms = MillisecondsBetween(c.sent_at, c.first_token_at);
    }
    if (c.phase == ReplyPhase::kActing) {
        AcceptAction(c, token, events);
        return;
    }
    if (std::ranges::find(stop_tokens_, token) != stop_tokens_.end()) {
        if (c.options.acts) {
            StartAction(c, events);
        } else {
            Finish(c, false, events);
        }
        return;
    }
    Emit(c, c.decoder->Push(token), events, drafted);
    ++c.stats.reply_tokens;
    ++stats_.generated_tokens;
    Append(c, std::span(&token, 1));
    if (c.stats.reply_tokens >= c.options.max_reply_tokens) {
        c.stats.truncated = true;
        if (c.options.acts) {
            StartAction(c, events);
        } else {
            Finish(c, true, events);
        }
    }
}

// The model has finished speaking: write the lead-in for it and hold what
// it writes next to the action grammar.
void Engine::StartAction(Conversation& c, std::vector<EngineEvent>& events) {
    Emit(c, c.decoder->Flush(), events);
    c.action_start = c.tokens.size();
    Append(c, actions_.lead_in);
    c.phase = ReplyPhase::kActing;
    c.grammar_state = actions_.grammar->Start();
    WriteForced(c, events);
}

// Adds one token of the action. A grammar with repeats allows actions of
// any length, but Send kept only kActionBudget tokens for one, so an action
// that reaches it is abandoned: the reply ends without one. Returns whether
// the action may go on.
bool Engine::AppendActionToken(Conversation& c, TokenId token, int next_state, bool forced,
                               std::vector<EngineEvent>& events) {
    Append(c, std::span(&token, 1));
    const std::string_view bytes = tokenizer_->TokenBytes(token);
    c.action += bytes;
    c.action_forced.append(bytes.size(), forced ? 1 : 0);
    c.grammar_state = next_state;
    if (c.tokens.size() - c.action_start >= static_cast<std::size_t>(kActionBudget)) {
        Finish(c, true, events);
        return false;
    }
    return true;
}

// A token of the action, sampled under the grammar's mask (so it is always
// allowed): advance through the grammar, then write whatever it forces.
void Engine::AcceptAction(Conversation& c, TokenId token, std::vector<EngineEvent>& events) {
    const int next = actions_.grammar->Next(c.grammar_state, token);
    if (next == ByteAutomaton::kDead) {
        throw std::logic_error("Engine: the sampler picked a token the action grammar forbids");
    }
    ++c.stats.action_tokens;
    ++stats_.action_tokens;
    ++stats_.generated_tokens;
    if (next == TokenGrammar::kFinished) {
        c.phase = ReplyPhase::kActed;
        Finish(c, c.stats.truncated, events);
        return;
    }
    if (AppendActionToken(c, token, next, false, events)) {
        WriteForced(c, events);
    }
}

// Jump-forward: while the grammar fixes what comes next, write it without
// running the model (it could only have written that). The forced tokens
// still go through the model with the next step, so its cache sees them,
// together with whatever it then chooses.
void Engine::WriteForced(Conversation& c, std::vector<EngineEvent>& events) {
    if (!options_.jump_forward) {
        return;
    }
    for (std::span<const TokenId> forced = actions_.grammar->Forced(c.grammar_state); !forced.empty();
         forced = actions_.grammar->Forced(c.grammar_state)) {
        for (const TokenId token : forced) {
            ++c.stats.forced_tokens;
            ++stats_.forced_tokens;
            const int next = actions_.grammar->Next(c.grammar_state, token);
            if (next == TokenGrammar::kFinished) {
                c.phase = ReplyPhase::kActed;
                Finish(c, c.stats.truncated, events);
                return;
            }
            if (!AppendActionToken(c, token, next, true, events)) {
                return;
            }
        }
    }
}

// The request that samples `c`'s next token: its settings and draw, and for
// a conversation that acts, a mask: while it speaks, no line breaks or "("; while
// it writes its action, the grammar's allowed tokens (with no repetition
// penalty: an action may well repeat a name it just said).
SamplingRequest Engine::SamplingFor(const Conversation& c, double draw, std::span<const TokenId> penalty_ids) const {
    SamplingRequest request{.params = c.options.sampling, .unit_draw = draw, .penalty_ids = penalty_ids};
    if (c.options.acts && c.phase == ReplyPhase::kSpeaking) {
        request.allowed = speech_mask_.Data();
    } else if (c.phase == ReplyPhase::kActing) {
        request.params.repetition_penalty = 1.0f;
        request.penalty_ids = {};
        request.allowed = actions_.grammar->DeviceMask(c.grammar_state);
    }
    return request;
}

// Sets the grammar of actions for conversations that act. Its end token
// must be the chat's end-of-turn token: an action ends the reply.
void Engine::SetActionLanguage(ActionLanguage language) {
    if (!language.grammar || language.lead_in.empty() || language.grammar->VocabSize() != tokenizer_->VocabSize()) {
        throw std::invalid_argument("Engine::SetActionLanguage: needs a grammar over this tokenizer and a lead-in");
    }
    for (const auto& [id, c] : conversations_) {
        if (c.phase == ReplyPhase::kActing) {   // its grammar state belongs to the old grammar
            throw std::logic_error("Engine::SetActionLanguage: a conversation is writing an action");
        }
    }
    actions_ = std::move(language);
    const int vocab = tokenizer_->VocabSize();
    std::vector<std::uint32_t> mask(static_cast<std::size_t>(vocab + 31) / 32, 0);
    for (TokenId id = 0; id < vocab; ++id) {
        const bool stop = std::ranges::find(stop_tokens_, id) != stop_tokens_.end();
        const std::string_view bytes = tokenizer_->TokenBytes(id);
        if (stop || (!tokenizer_->IsSpecial(id) && !bytes.contains('\n') && !bytes.contains('('))) {
            mask[id / 32] |= 1u << (id % 32);
        }
    }
    speech_mask_ = DeviceBuffer<std::uint32_t>(mask.size());
    speech_mask_.CopyFromHost(mask);
}

// How many tokens the Speculator guesses per step from now on (0: none).
void Engine::SetDraftTokens(int draft_tokens) {
    if (draft_tokens < 0) {
        throw std::invalid_argument("Engine::SetDraftTokens: negative");
    }
    options_.draft_tokens = draft_tokens;
}

// Closes the turn (the end-of-turn marker and the newline after it go in
// with the next message, as in ChatSession) and reports the reply's stats.
void Engine::Finish(Conversation& c, bool truncated, std::vector<EngineEvent>& events) {
    if (c.decoder) {
        Emit(c, c.decoder->Flush(), events);
    }
    const TokenId end_of_turn = template_.EndOfTurn();
    Append(c, std::span(&end_of_turn, 1));
    Append(c, template_.EncodeAfterReply());
    c.stats.truncated = truncated;
    c.stats.prompt_tokens = c.prompt_tokens;
    if (c.phase == ReplyPhase::kActed) {   // a cancelled or abandoned action doesn't count
        ++stats_.actions;
        events.push_back(
            {.conversation = c.id, .kind = EngineEventKind::kAction, .text = c.action, .forced = c.action_forced});
    }
    c.phase = ReplyPhase::kSpeaking;
    c.stats.decode_ms = c.stats.reply_tokens > 0 ? MillisecondsBetween(c.first_token_at, Clock::now()) : 0.0;
    c.stats.decode_bytes_per_token = static_cast<double>(transformer_->DecodeBytes(c.kv.length));
    c.stats.peak_bandwidth = PeakMemoryBandwidth();
    c.replying = false;
    c.cancelled = false;
    c.decoder.reset();
    events.push_back({.conversation = c.id, .kind = EngineEventKind::kReplyFinished, .stats = c.stats});
}

void Engine::Emit(Conversation& c, std::string text, std::vector<EngineEvent>& events, bool drafted) {
    if (!text.empty()) {
        events.push_back(
            {.conversation = c.id, .kind = EngineEventKind::kText, .text = std::move(text), .drafted = drafted});
    }
}

// Drops the turns before turn_starts[first_kept_turn] (all of them when it's
// past the end), keeping the system prompt. Positions change, so the KV
// cache can't be kept: it is released and the kept tokens run again.
void Engine::Rebuild(Conversation& c, std::size_t first_kept_turn) {
    const std::size_t keep_from = first_kept_turn < c.turn_starts.size() ? c.turn_starts[first_kept_turn] : c.tokens.size();
    std::vector<TokenId> kept(c.tokens.begin(), c.tokens.begin() + c.system_tokens);
    kept.insert(kept.end(), c.tokens.begin() + keep_from, c.tokens.end());
    std::vector<std::size_t> starts;
    for (std::size_t i = first_kept_turn; i < c.turn_starts.size(); ++i) {
        starts.push_back(c.turn_starts[i] - keep_from + c.system_tokens);
    }
    stats_.forgotten_turns += static_cast<std::int64_t>(first_kept_turn);
    Cache().Release(c.kv);
    c.published_pages = 0;
    c.tokens.clear();
    c.distinct.clear();
    c.seen.clear();
    Append(c, kept);
    c.turn_starts = std::move(starts);
}

// Appends tokens to the conversation (not yet cached) and to its distinct
// set for the repetition penalty.
void Engine::Append(Conversation& c, std::span<const TokenId> ids) {
    c.tokens.insert(c.tokens.end(), ids.begin(), ids.end());
    for (const TokenId id : ids) {
        if (c.seen.insert(id).second) {
            c.distinct.push_back(id);
        }
    }
}

Engine::Conversation& Engine::Get(ConversationId id) {
    const auto it = conversations_.find(id);
    if (it == conversations_.end()) {
        throw std::invalid_argument("Engine: unknown conversation");
    }
    return it->second;
}

// A snapshot for the X-ray view: who owns each page, the last batch, totals.
EngineStats Engine::Stats() const {
    EngineStats stats = stats_;
    KvCacheManager& cache = transformer_->KvCache();
    stats.page_owner.assign(cache.PageCount(), EngineStats::kFree);
    stats.page_shared.assign(cache.PageCount(), 0);
    for (PageId page = 0; page < cache.PageCount(); ++page) {
        const int owners = cache.RefCount(page);
        if (owners > 0) {
            stats.page_owner[page] = EngineStats::kCached;
            stats.page_shared[page] = owners > 2 || (owners == 2 && !cache.IsCached(page)) ? 1 : 0;
        }
    }
    for (const auto& [id, c] : conversations_) {
        for (const PageId page : c.kv.pages) {
            stats.page_owner[page] = id;
        }
    }
    stats.draft_tokens = speculator_ != nullptr ? options_.draft_tokens : 0;
    stats.pages_free = cache.FreePages();
    stats.pages_cached = cache.CachedPages();
    stats.page_tokens = cache.PageSize();
    stats.max_batch = transformer_->Options().max_batch;
    stats.max_speculative_batch = options_.max_speculative_batch;
    stats.weight_format = transformer_->Weights().format;
    stats.kv_format = transformer_->Options().kv_format;
    stats.weight_bytes = transformer_->WeightBytes();
    stats.kv_bytes = transformer_->Options().kv_cache_bytes;
    return stats;
}

// Swaps in other weights and a KV cache of another size or format, between
// steps. The old cache's pages go with it, so every conversation is evicted
// first: it recomputes its history the next time it runs, and a reply in
// progress carries on from where it was.
void Engine::Reconfigure(ModelWeights weights, std::size_t kv_bytes, KvFormat kv_format) {
    transformer_->CheckKvCache(kv_bytes, kv_format);   // fails before anything changes
    const std::int64_t evictions = stats_.evictions;
    for (auto& [id, c] : conversations_) {
        if (!c.kv.pages.empty()) {
            Evict(c);
        }
    }
    stats_.evictions = evictions;   // a format switch isn't memory pressure
    transformer_->ReplaceWeights(std::move(weights));
    transformer_->ResizeKvCache(kv_bytes, kv_format);
}

}  // namespace murmur
