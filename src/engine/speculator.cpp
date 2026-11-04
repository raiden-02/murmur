#include "engine/speculator.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace murmur {

namespace {

// New tokens up to which a draft pass treats them as decode rows (the main
// model's token plus at most one unfed guess, usually).
constexpr std::size_t kCatchUpRows = 4;

}  // namespace

Speculator::Speculator(std::unique_ptr<Transformer> draft, int candidate_count, std::span<const TokenId> stop_tokens)
    : draft_(std::move(draft)),
      candidate_count_(candidate_count),
      stop_tokens_(stop_tokens.begin(), stop_tokens.end()),
      sampler_(draft_->Options().max_batch, draft_->Options().max_batch * draft_->Options().max_context) {
    if (candidate_count <= 0 || candidate_count > draft_->Config().vocab_size) {
        throw std::invalid_argument("Speculator: the tokenizer doesn't fit the draft model's vocabulary");
    }
    // A catch-up pass asks for logits on up to kCatchUpRows rows of every
    // sequence in the batch (see Run).
    const TransformerOptions& limits = draft_->Options();
    if (limits.max_logit_rows < static_cast<int>(kCatchUpRows) * limits.max_batch) {
        throw std::invalid_argument("Speculator: the draft model needs max_logit_rows >= 4 x max_batch");
    }
}

// Guesses, for each request, one token per draw (fewer if the guess ends the
// reply or the draft cache has no room). The requests' guesses are made
// together: one pass of the draft model per guessed position, for all
// conversations at once. The first pass also catches up on what the
// conversation added since the last call (the main model's own token, and
// the last guess, which is never fed back when it's guessed), as decode
// rows, so it replays as a CUDA graph like every later pass.
std::vector<std::vector<TokenId>> Speculator::Propose(std::span<const DraftRequest> requests) {
    if (requests.size() > static_cast<std::size_t>(draft_->Options().max_batch)) {
        throw std::invalid_argument("Speculator::Propose: more conversations than the draft model's max_batch");
    }
    ++calls_;
    std::vector<std::vector<TokenId>> guesses(requests.size());
    std::vector<std::size_t> active;
    std::vector<std::vector<TokenId>> penalties(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const DraftRequest& r = requests[i];
        if (r.draws.empty() || r.tokens.empty()) {
            continue;
        }
        Cached& cached = cached_[r.conversation];
        cached.last_used = calls_;
        // Room for the conversation plus every guess but the last (which is never fed back).
        const int length = static_cast<int>(r.tokens.size() + r.draws.size()) - 1;
        if (length > draft_->Options().max_context || !Prepare(cached, r.tokens, length)) {
            continue;
        }
        penalties[i].assign(r.penalty_ids.begin(), r.penalty_ids.end());
        active.push_back(i);
    }

    // A long catch-up (the first guess in a conversation, or after it forgot
    // turns) runs as prompt chunks first, one conversation at a time.
    const int chunk = draft_->Options().max_chunk;
    for (const std::size_t i : active) {
        Cached& cached = cached_[requests[i].conversation];
        const std::span<const TokenId> all = requests[i].tokens;
        while (all.size() - cached.tokens.size() > kCatchUpRows) {
            const std::size_t count = std::min<std::size_t>(chunk, all.size() - kCatchUpRows - cached.tokens.size());
            Cached* one[] = {&cached};
            const std::span<const TokenId> part = all.subspan(cached.tokens.size(), count);
            Run(one, std::span(&part, 1));
        }
    }

    // Then one position at a time for every conversation together: feed what
    // the draft hasn't seen (the newest tokens, then each guess), guess the next.
    std::vector<TokenId> fed(requests.size());
    for (std::size_t position = 0; !active.empty(); ++position) {
        std::vector<Cached*> sequences;
        std::vector<std::span<const TokenId>> tokens;
        std::vector<int> last_rows;
        int rows = 0;
        for (const std::size_t i : active) {
            Cached& cached = cached_[requests[i].conversation];
            if (position == 0) {
                tokens.push_back(requests[i].tokens.subspan(cached.tokens.size()));
            } else {
                fed[i] = guesses[i].back();
                tokens.push_back(std::span(&fed[i], 1));
            }
            sequences.push_back(&cached);
            rows += static_cast<int>(tokens.back().size());
            last_rows.push_back(rows - 1);
        }
        Run(sequences, tokens);

        std::vector<SamplingRequest> sampling;
        for (std::size_t a = 0; a < active.size(); ++a) {
            const std::size_t i = active[a];
            sampling.push_back({.params = requests[i].sampling,
                                .unit_draw = requests[i].draws[position],
                                .penalty_ids = penalties[i],
                                .logits_row = last_rows[a]});
        }
        const std::span<const TokenId> next = sampler_.Sample(draft_->StepLogits(), draft_->Config().vocab_size,
                                                              candidate_count_, sampling, draft_->Stream());
        std::vector<std::size_t> still;
        for (std::size_t a = 0; a < active.size(); ++a) {
            const std::size_t i = active[a];
            guesses[i].push_back(next[a]);
            if (std::ranges::find(penalties[i], next[a]) == penalties[i].end()) {
                penalties[i].push_back(next[a]);
            }
            const bool ends_reply = std::ranges::find(stop_tokens_, next[a]) != stop_tokens_.end();
            if (!ends_reply && guesses[i].size() < requests[i].draws.size()) {
                still.push_back(i);
            }
        }
        active = std::move(still);
    }
    return guesses;
}

// Drops a conversation's draft cache (it closed).
void Speculator::Forget(int conversation) {
    const auto it = cached_.find(conversation);
    if (it != cached_.end()) {
        draft_->KvCache().Release(it->second.kv);
        cached_.erase(it);
    }
}

// Keeps what the draft cache already holds of `tokens` (its longest common
// prefix with them, never the newest token, which must run to give logits),
// and reserves pages for `length` positions, dropping the least recently
// used conversations' draft caches (not ones guessing in this call) if there
// is no room.
bool Speculator::Prepare(Cached& cached, std::span<const TokenId> tokens, int length) {
    const auto [mine, theirs] = std::ranges::mismatch(cached.tokens, tokens.first(tokens.size() - 1));
    const int keep = static_cast<int>(mine - cached.tokens.begin());
    if (keep < cached.kv.length) {
        draft_->KvCache().Truncate(cached.kv, keep);
    }
    cached.tokens.resize(keep);
    KvCacheManager& cache = draft_->KvCache();
    while (!cache.Reserve(cached.kv, length)) {
        auto oldest = cached_.end();
        for (auto it = cached_.begin(); it != cached_.end(); ++it) {
            // Never a conversation guessing in this same call.
            if (it->second.last_used != calls_ && !it->second.kv.pages.empty() &&
                (oldest == cached_.end() || it->second.last_used < oldest->second.last_used)) {
                oldest = it;
            }
        }
        if (oldest == cached_.end()) {
            return false;
        }
        cache.Release(oldest->second.kv);
        oldest->second.tokens.clear();
    }
    return true;
}

// One draft pass: each sequence takes its tokens. Up to kCatchUpRows of
// them run as decode rows with logits for every one (StepLogits() then holds
// each sequence's rows in turn); a longer run is a prompt chunk with logits
// for its last token only.
void Speculator::Run(std::span<Cached* const> sequences, std::span<const std::span<const TokenId>> tokens) {
    std::vector<StepEntry> entries;
    for (std::size_t s = 0; s < sequences.size(); ++s) {
        const int count = static_cast<int>(tokens[s].size());
        const bool decode_rows = count <= kCatchUpRows;
        entries.push_back({.sequence = &sequences[s]->kv,
                           .tokens = tokens[s],
                           .logits = decode_rows ? count : 1,
                           .decode_each_token = decode_rows});
    }
    draft_->Step(entries);
    for (std::size_t s = 0; s < sequences.size(); ++s) {
        sequences[s]->kv.length += static_cast<int>(tokens[s].size());
        sequences[s]->tokens.insert(sequences[s]->tokens.end(), tokens[s].begin(), tokens[s].end());
    }
}

}  // namespace murmur
