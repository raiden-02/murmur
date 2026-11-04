#pragma once

#include "common/device_buffer.hpp"
#include "model/tokenizer.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace murmur {

// A regular language over bytes, compiled from a small grammar into a
// deterministic finite automaton (DFA): from any state, each byte leads to
// at most one next state, so checking a string is a walk, byte by byte.
//
// Grammar text, one rule per line (the first rule is the whole language;
// lines starting with # are comments):
//   action ::= "walk(" place ")" | "none"
//   place  ::= "DOCKS" | "BAR"
//   price  ::= [1-9] [0-9]{0,2}
// "..." is literal text (with \" \\ \n escapes), [...] a set of bytes (with
// ranges like a-z), names refer to other rules, juxtaposition is a sequence,
// | separates alternatives, ( ) groups, and ? * + {m} {m,n} repeat. Rules may
// not refer to themselves, directly or not: the language must stay regular.
class ByteAutomaton {
public:
    static constexpr int kDead = -1;

    static ByteAutomaton Compile(std::string_view grammar);

    int Start() const { return 0; }
    int Next(int state, unsigned char byte) const { return next_[static_cast<std::size_t>(state) * 256 + byte]; }
    int Walk(int state, std::string_view bytes) const;
    bool Accepting(int state) const { return accepting_[state] != 0; }
    int StateCount() const { return static_cast<int>(accepting_.size()); }

private:
    std::vector<int> next_;                // state * 256 + byte -> state, or kDead
    std::vector<std::uint8_t> accepting_;  // per state
};

// A ByteAutomaton lifted to the model's tokens: for every state, which
// tokens may come next (their bytes all keep the automaton alive) and where
// each leads. The end-of-text token is allowed exactly in accepting states.
// Each state's allowed tokens are also a bitmask in GPU memory, which the
// sampler applies to the logits: the model can only write what the grammar
// allows. Where the grammar fixes the next bytes (the "(" after "walk"),
// Forced() has them as tokens, for the engine to write without the model,
// except a last token the model could merge with its next choice.
class TokenGrammar {
public:
    static constexpr int kFinished = -2;   // the state after the end token

    TokenGrammar(const ByteAutomaton& automaton, const Tokenizer& tokenizer, TokenId end_token);

    int Start() const { return start_; }
    int Next(int state, TokenId token) const;
    bool Accepting(int state) const { return automaton_accepting_[state] != 0; }
    std::span<const TokenId> Forced(int state) const { return forced_[state]; }
    std::size_t AllowedCount(int state) const { return transitions_[state].size(); }
    const std::uint32_t* DeviceMask(int state) const;
    int StateCount() const { return static_cast<int>(transitions_.size()); }
    int VocabSize() const { return vocab_size_; }

private:
    int start_ = 0;
    int vocab_size_ = 0;
    int mask_words_ = 0;
    std::vector<std::vector<std::pair<TokenId, int>>> transitions_;   // per state, sorted by token
    std::vector<std::uint8_t> automaton_accepting_;
    std::vector<std::vector<TokenId>> forced_;   // per state: the tokens the grammar leaves no choice about
    DeviceBuffer<std::uint32_t> masks_;   // StateCount() x mask_words_ bits
};

}  // namespace murmur
