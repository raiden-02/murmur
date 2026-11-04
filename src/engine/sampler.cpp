#include "engine/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace murmur {

namespace {

struct Candidate {
    float logit;
    TokenId id;
};

void Validate(const SamplingParams& p) {
    if (!(p.temperature >= 0.0f) || p.top_k < 0 || !(p.top_p > 0.0f && p.top_p <= 1.0f) ||
        !(p.repetition_penalty > 0.0f)) {
        throw std::invalid_argument("SamplingParams: out of range");
    }
}

// Penalizes every token already in the conversation once: positive logits
// are divided by the penalty and negative ones multiplied, so both move
// towards "less likely".
void ApplyRepetitionPenalty(std::span<float> logits, std::span<const TokenId> history, float penalty) {
    if (penalty == 1.0f) {
        return;
    }
    std::vector<bool> seen(logits.size(), false);
    for (const TokenId id : history) {
        if (id < 0 || static_cast<std::size_t>(id) >= logits.size() || seen[id]) {
            continue;
        }
        seen[id] = true;
        logits[id] = logits[id] < 0.0f ? logits[id] * penalty : logits[id] / penalty;
    }
}

// The k highest logits, highest first (all of them when k is 0).
std::vector<Candidate> TopK(std::span<const float> logits, int k) {
    std::vector<Candidate> candidates(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        candidates[i] = {logits[i], static_cast<TokenId>(i)};
    }
    const auto by_logit_desc = [](const Candidate& a, const Candidate& b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    };
    const std::size_t keep = k == 0 ? candidates.size() : std::min<std::size_t>(k, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(), by_logit_desc);
    candidates.resize(keep);
    return candidates;
}

// Softmax of the candidates' logits (already sorted highest first).
std::vector<double> Probabilities(const std::vector<Candidate>& candidates) {
    const double max_logit = candidates.front().logit;
    std::vector<double> probabilities(candidates.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        probabilities[i] = std::exp(static_cast<double>(candidates[i].logit) - max_logit);
        sum += probabilities[i];
    }
    for (double& p : probabilities) {
        p /= sum;
    }
    return probabilities;
}

// Keeps the smallest set of most-likely tokens whose probability mass
// reaches top_p ("nucleus" sampling), always at least one. Matches Hugging
// Face: walking from the least likely token, drop tokens while the dropped
// mass stays within 1 - top_p.
std::size_t NucleusSize(const std::vector<double>& probabilities, float top_p) {
    double dropped = 0.0;
    std::size_t keep = probabilities.size();
    while (keep > 1 && dropped + probabilities[keep - 1] <= 1.0 - top_p) {
        dropped += probabilities[keep - 1];
        --keep;
    }
    return keep;
}

}  // namespace

Sampler::Sampler(SamplingParams params) : params_(params), rng_(params.seed) {
    Validate(params_);
}

// Picks the next token with this sampler's own random numbers. `logits`
// holds one score per candidate token (the caller passes only ids the
// tokenizer can decode) and is modified in place; `history` is every token
// so far, for the repetition penalty.
TokenId Sampler::Sample(std::span<float> logits, std::span<const TokenId> history) {
    return Pick(logits, history, params_.temperature == 0.0f ? 0.0 : NextDraw());
}

// The next random number in [0, 1) of this sampler's sequence (its seed
// makes runs repeatable).
double Sampler::NextDraw() {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
}

// Picks the next token given the random number `unit_draw` in [0, 1).
// Steps, in Hugging Face's order: repetition penalty, temperature, top-k,
// top-p, then the draw, weighted by the remaining probabilities. The GPU
// sampler (SampleRows) follows exactly these steps.
TokenId Sampler::Pick(std::span<float> logits, std::span<const TokenId> history, double unit_draw) const {
    if (logits.empty() || !(unit_draw >= 0.0 && unit_draw < 1.0)) {
        throw std::invalid_argument("Sampler::Pick: no logits, or a draw outside [0, 1)");
    }
    ApplyRepetitionPenalty(logits, history, params_.repetition_penalty);
    if (params_.temperature == 0.0f) {
        return static_cast<TokenId>(std::ranges::max_element(logits) - logits.begin());
    }
    for (float& logit : logits) {
        logit /= params_.temperature;
    }
    std::vector<Candidate> candidates = TopK(logits, params_.top_k);
    std::vector<double> probabilities = Probabilities(candidates);
    const std::size_t keep = NucleusSize(probabilities, params_.top_p);

    const double kept_mass = std::accumulate(probabilities.begin(), probabilities.begin() + keep, 0.0);
    double draw = unit_draw * kept_mass;
    for (std::size_t i = 0; i + 1 < keep; ++i) {
        draw -= probabilities[i];
        if (draw < 0.0) {
            return candidates[i].id;
        }
    }
    return candidates[keep - 1].id;
}

}  // namespace murmur
