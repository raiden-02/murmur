#include "engine/grammar.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cctype>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace murmur {

namespace {

using ByteSet = std::bitset<256>;

// Limits that keep a grammar's cost bounded. Every DFA state gets a mask
// over the whole vocabulary on the host and the GPU (~19 KB for 151,936
// tokens), and the market's grammar needs 913 states. Nested repeats grow
// the NFA exponentially, so its size is capped as it's built.
constexpr int kMaxDfaStates = 4096;
constexpr int kMaxNfaStates = 1 << 16;
constexpr int kMaxRepeat = 64;

// The grammar's syntax tree: what one rule's right-hand side says.
struct Node {
    enum class Kind { kLiteral, kSet, kRule, kSequence, kChoice, kRepeat };
    Kind kind = Kind::kSequence;
    std::string text;   // kLiteral: the bytes; kRule: the rule's name
    ByteSet set;        // kSet
    std::vector<Node> children;
    int min = 0;        // kRepeat
    int max = 0;        // kRepeat; -1: no limit
};

[[noreturn]] void Fail(const std::string& message) {
    throw std::invalid_argument("grammar: " + message);
}

// Recursive descent over one rule's right-hand side:
//   choice   := sequence ("|" sequence)*
//   sequence := repeated*
//   repeated := atom ("?" | "*" | "+" | "{" m ("," n?)? "}")?
//   atom     := string | set | name | "(" choice ")"
class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Node ParseAll() {
        Node node = Choice();
        Skip();
        if (at_ != text_.size()) {
            Fail("unexpected '" + std::string(1, text_[at_]) + "' in \"" + std::string(text_) + "\"");
        }
        return node;
    }

private:
    Node Choice() {
        Node choice{.kind = Node::Kind::kChoice};
        choice.children.push_back(Sequence());
        while (Take('|')) {
            choice.children.push_back(Sequence());
        }
        return choice.children.size() == 1 ? std::move(choice.children.front()) : std::move(choice);
    }

    Node Sequence() {
        Node sequence{.kind = Node::Kind::kSequence};
        Skip();
        while (at_ < text_.size() && text_[at_] != '|' && text_[at_] != ')') {
            sequence.children.push_back(Repeated());
            Skip();
        }
        if (sequence.children.empty()) {
            Fail("empty alternative in \"" + std::string(text_) + "\"");
        }
        return sequence.children.size() == 1 ? std::move(sequence.children.front()) : std::move(sequence);
    }

    Node Repeated() {
        Node atom = Atom();
        int min = 1;
        int max = 1;
        if (Take('?')) {
            min = 0;
        } else if (Take('*')) {
            min = 0;
            max = -1;
        } else if (Take('+')) {
            max = -1;
        } else if (Take('{')) {
            min = max = Number();
            if (Take(',')) {
                Skip();
                max = at_ < text_.size() && text_[at_] == '}' ? -1 : Number();
            }
            if (!Take('}') || min > kMaxRepeat || max > kMaxRepeat || (max != -1 && max < min)) {
                Fail("bad {m,n} repetition");
            }
        } else {
            return atom;
        }
        Node repeat{.kind = Node::Kind::kRepeat, .min = min, .max = max};
        repeat.children.push_back(std::move(atom));
        return repeat;
    }

    Node Atom() {
        Skip();
        if (at_ >= text_.size()) {
            Fail("expression ends too early");
        }
        const char c = text_[at_];
        if (c == '"') {
            ++at_;
            Node literal{.kind = Node::Kind::kLiteral};
            while (at_ < text_.size() && text_[at_] != '"') {
                literal.text += Escaped();
            }
            if (!Take('"') || literal.text.empty()) {
                Fail("unterminated or empty string");
            }
            return literal;
        }
        if (c == '[') {
            ++at_;
            Node set{.kind = Node::Kind::kSet};
            while (at_ < text_.size() && text_[at_] != ']') {
                const unsigned char low = static_cast<unsigned char>(Escaped());
                unsigned char high = low;
                if (at_ + 1 < text_.size() && text_[at_] == '-' && text_[at_ + 1] != ']') {
                    ++at_;
                    high = static_cast<unsigned char>(Escaped());
                }
                if (high < low) {
                    Fail("bad range in a [set]");
                }
                for (int b = low; b <= high; ++b) {
                    set.set.set(b);
                }
            }
            if (!Take(']') || set.set.none()) {
                Fail("unterminated or empty [set]");
            }
            return set;
        }
        if (c == '(') {
            ++at_;
            Node inner = Choice();
            if (!Take(')')) {
                Fail("missing ')'");
            }
            return inner;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            Node rule{.kind = Node::Kind::kRule};
            while (at_ < text_.size() && (std::isalnum(static_cast<unsigned char>(text_[at_])) || text_[at_] == '_')) {
                rule.text += text_[at_++];
            }
            return rule;
        }
        Fail("unexpected '" + std::string(1, c) + "'");
    }

    char Escaped() {
        char c = text_[at_++];
        if (c != '\\') {
            return c;
        }
        if (at_ >= text_.size()) {
            Fail("dangling escape");
        }
        c = text_[at_++];
        return c == 'n' ? '\n' : c == 't' ? '\t' : c;
    }

    int Number() {
        Skip();
        int value = 0;
        bool any = false;
        while (at_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[at_]))) {
            value = value * 10 + (text_[at_++] - '0');
            any = true;
            if (value > kMaxRepeat) {
                Fail("repetition count too large");
            }
        }
        if (!any) {
            Fail("expected a number");
        }
        return value;
    }

    bool Take(char c) {
        Skip();
        if (at_ < text_.size() && text_[at_] == c) {
            ++at_;
            return true;
        }
        return false;
    }

    void Skip() {
        while (at_ < text_.size() && (text_[at_] == ' ' || text_[at_] == '\t')) {
            ++at_;
        }
    }

    std::string_view text_;
    std::size_t at_ = 0;
};

// A nondeterministic automaton built piece by piece (Thompson's
// construction): every syntax node becomes a fragment with one way in and
// one way out, joined by empty (epsilon) moves.
class NfaBuilder {
public:
    struct State {
        std::vector<int> epsilon;
        ByteSet on;    // bytes that move to `to`
        int to = -1;
    };
    struct Fragment {
        int in;
        int out;
    };

    explicit NfaBuilder(const std::map<std::string, Node, std::less<>>& rules) : rules_(rules) {}

    Fragment Build(const Node& node) {
        switch (node.kind) {
        case Node::Kind::kLiteral: {
            const int in = Add();
            int at = in;
            for (const char c : node.text) {
                const int next = Add();
                states_[at].on.set(static_cast<unsigned char>(c));
                states_[at].to = next;
                at = next;
            }
            return {in, at};
        }
        case Node::Kind::kSet: {
            const int in = Add();
            const int out = Add();
            states_[in].on = node.set;
            states_[in].to = out;
            return {in, out};
        }
        case Node::Kind::kRule: {
            const auto it = rules_.find(node.text);
            if (it == rules_.end()) {
                Fail("unknown rule \"" + node.text + "\"");
            }
            if (!expanding_.insert(node.text).second) {
                Fail("rule \"" + node.text + "\" refers to itself: only regular grammars are supported");
            }
            const Fragment fragment = Build(it->second);
            expanding_.erase(node.text);
            return fragment;
        }
        case Node::Kind::kSequence: {
            const Fragment first = Build(node.children.front());
            int out = first.out;
            for (std::size_t i = 1; i < node.children.size(); ++i) {
                const Fragment next = Build(node.children[i]);
                states_[out].epsilon.push_back(next.in);
                out = next.out;
            }
            return {first.in, out};
        }
        case Node::Kind::kChoice: {
            const int in = Add();
            const int out = Add();
            for (const Node& child : node.children) {
                const Fragment option = Build(child);
                states_[in].epsilon.push_back(option.in);
                states_[option.out].epsilon.push_back(out);
            }
            return {in, out};
        }
        case Node::Kind::kRepeat: {
            // min required copies, then (max - min) optional ones, or a loop.
            const int in = Add();
            int out = in;
            for (int i = 0; i < node.min; ++i) {
                const Fragment copy = Build(node.children.front());
                states_[out].epsilon.push_back(copy.in);
                out = copy.out;
            }
            if (node.max == -1) {
                const Fragment loop = Build(node.children.front());
                const int end = Add();
                states_[out].epsilon.push_back(loop.in);
                states_[out].epsilon.push_back(end);
                states_[loop.out].epsilon.push_back(loop.in);
                states_[loop.out].epsilon.push_back(end);
                return {in, end};
            }
            const int end = Add();
            for (int i = node.min; i < node.max; ++i) {
                const Fragment copy = Build(node.children.front());
                states_[out].epsilon.push_back(copy.in);
                states_[out].epsilon.push_back(end);
                out = copy.out;
            }
            states_[out].epsilon.push_back(end);
            return {in, end};
        }
        }
        Fail("unknown syntax node");
    }

    const std::vector<State>& States() const { return states_; }

private:
    int Add() {
        if (static_cast<int>(states_.size()) >= kMaxNfaStates) {
            Fail("the grammar is too large");
        }
        states_.emplace_back();
        return static_cast<int>(states_.size()) - 1;
    }

    const std::map<std::string, Node, std::less<>>& rules_;
    std::set<std::string> expanding_;
    std::vector<State> states_;
};

// The NFA states reachable from `seeds` by empty moves alone, sorted.
std::vector<int> Closure(const std::vector<NfaBuilder::State>& nfa, std::vector<int> seeds) {
    std::vector<std::uint8_t> seen(nfa.size(), 0);
    std::vector<int> stack = std::move(seeds);
    std::vector<int> closure;
    while (!stack.empty()) {
        const int s = stack.back();
        stack.pop_back();
        if (seen[s] != 0) {
            continue;
        }
        seen[s] = 1;
        closure.push_back(s);
        for (const int t : nfa[s].epsilon) {
            stack.push_back(t);
        }
    }
    std::ranges::sort(closure);
    return closure;
}

}  // namespace

// Parses the rules, builds the nondeterministic automaton of the first rule
// (rule references expanded in place), and turns it into a deterministic
// one by the subset construction: each DFA state is the set of NFA states
// the input so far could be in. Throws std::invalid_argument for a syntax
// error, an unknown or recursive rule, or a language too large to expand.
ByteAutomaton ByteAutomaton::Compile(std::string_view grammar) {
    std::map<std::string, Node, std::less<>> rules;
    std::string first;
    std::size_t begin = 0;
    while (begin < grammar.size()) {
        const std::size_t end = std::min(grammar.find('\n', begin), grammar.size());
        std::string_view line = grammar.substr(begin, end - begin);
        begin = end + 1;
        const std::size_t first_char = line.find_first_not_of(" \t\r");
        if (first_char == std::string_view::npos || line[first_char] == '#') {
            continue;   // a blank line or a comment
        }
        const std::size_t arrow = line.find("::=");
        if (arrow == std::string_view::npos) {
            Fail("expected \"name ::= expression\", got \"" + std::string(line) + "\"");
        }
        std::string name(line.substr(0, arrow));
        std::erase_if(name, [](char c) { return c == ' ' || c == '\t'; });
        if (name.empty() || rules.contains(name)) {
            Fail("missing or repeated rule name \"" + name + "\"");
        }
        std::string_view body = line.substr(arrow + 3);
        while (!body.empty() && (body.back() == '\r' || body.back() == ' ')) {
            body.remove_suffix(1);
        }
        rules.emplace(name, Parser(body).ParseAll());
        if (first.empty()) {
            first = name;
        }
    }
    if (first.empty()) {
        Fail("no rules");
    }

    NfaBuilder builder(rules);
    const NfaBuilder::Fragment root = builder.Build(Node{.kind = Node::Kind::kRule, .text = first});
    const std::vector<NfaBuilder::State>& nfa = builder.States();

    ByteAutomaton dfa;
    std::map<std::vector<int>, int> ids;
    std::vector<std::vector<int>> sets;
    const auto intern = [&](std::vector<int> set) {
        const auto [it, inserted] = ids.emplace(std::move(set), static_cast<int>(sets.size()));
        if (inserted) {
            if (static_cast<int>(sets.size()) >= kMaxDfaStates) {
                Fail("the language needs too many states");
            }
            sets.push_back(it->first);
            dfa.accepting_.push_back(std::ranges::binary_search(it->first, root.out) ? 1 : 0);
            dfa.next_.resize(sets.size() * 256, kDead);
        }
        return it->second;
    };
    intern(Closure(nfa, {root.in}));
    for (std::size_t s = 0; s < sets.size(); ++s) {
        const std::vector<int> current = sets[s];   // interning may grow `sets`
        for (int byte = 0; byte < 256; ++byte) {
            std::vector<int> moved;
            for (const int n : current) {
                if (nfa[n].on.test(byte)) {
                    moved.push_back(nfa[n].to);
                }
            }
            if (!moved.empty()) {
                const int target = intern(Closure(nfa, std::move(moved)));
                dfa.next_[s * 256 + byte] = target;
            }
        }
    }
    return dfa;
}

// The state after reading `bytes` from `state`, or kDead if the language
// doesn't allow them.
int ByteAutomaton::Walk(int state, std::string_view bytes) const {
    for (const char c : bytes) {
        state = Next(state, static_cast<unsigned char>(c));
        if (state == kDead) {
            return kDead;
        }
    }
    return state;
}

// For every automaton state, finds the tokens that can follow by walking
// each candidate's bytes (only tokens whose first byte the state accepts are
// tried, so most states look at a handful), then uploads one bitmask of
// allowed tokens per state. Special tokens (<|im_end|> and the like) are
// never matched by their spelling; `end_token` alone ends the text, in
// accepting states.
namespace {

// The bytes the automaton fixes from `state`: as long as exactly one byte
// leads on and the text so far isn't a whole sentence (where the end could
// come instead), that byte is the only possible next one.
std::string ForcedBytes(const ByteAutomaton& automaton, int state) {
    constexpr std::size_t kMaxForced = 256;   // a cycle of single bytes never reaches an end
    std::string text;
    while (!automaton.Accepting(state) && text.size() < kMaxForced) {
        int only = -1;
        for (int byte = 0; byte < 256; ++byte) {
            if (automaton.Next(state, static_cast<unsigned char>(byte)) != ByteAutomaton::kDead) {
                if (only >= 0) {
                    return text;
                }
                only = byte;
            }
        }
        if (only < 0) {
            break;
        }
        text += static_cast<char>(only);
        state = automaton.Next(state, static_cast<unsigned char>(only));
    }
    return text;
}

}  // namespace

TokenGrammar::TokenGrammar(const ByteAutomaton& automaton, const Tokenizer& tokenizer, TokenId end_token)
    : start_(automaton.Start()), vocab_size_(tokenizer.VocabSize()), mask_words_((vocab_size_ + 31) / 32) {
    std::array<std::vector<TokenId>, 256> by_first_byte;
    for (TokenId id = 0; id < vocab_size_; ++id) {
        const std::string_view bytes = tokenizer.TokenBytes(id);
        if (!bytes.empty() && !tokenizer.IsSpecial(id)) {
            by_first_byte[static_cast<unsigned char>(bytes.front())].push_back(id);
        }
    }
    const int states = automaton.StateCount();
    transitions_.resize(states);
    automaton_accepting_.resize(states);
    std::vector<std::uint32_t> masks(static_cast<std::size_t>(states) * mask_words_, 0);
    for (int s = 0; s < states; ++s) {
        auto& allowed = transitions_[s];
        for (int byte = 0; byte < 256; ++byte) {
            if (automaton.Next(s, static_cast<unsigned char>(byte)) == ByteAutomaton::kDead) {
                continue;
            }
            for (const TokenId id : by_first_byte[byte]) {
                const int end = automaton.Walk(s, tokenizer.TokenBytes(id));
                if (end != ByteAutomaton::kDead) {
                    allowed.emplace_back(id, end);
                }
            }
        }
        automaton_accepting_[s] = automaton.Accepting(s) ? 1 : 0;
        if (automaton.Accepting(s)) {
            allowed.emplace_back(end_token, kFinished);
        }
        std::ranges::sort(allowed);
        for (const auto& [id, next] : allowed) {
            masks[static_cast<std::size_t>(s) * mask_words_ + id / 32] |= 1u << (id % 32);
        }
    }
    masks_ = DeviceBuffer<std::uint32_t>(masks.size());
    masks_.CopyFromHost(masks);

    // The fixed bytes, as tokens, but without a last token that the model
    // could write merged with what it chooses next: with " " fixed before
    // "walk", the model writes " walk", one token, and a lone " " would push
    // it off the way it writes.
    forced_.resize(states);
    for (int s = 0; s < states; ++s) {
        std::vector<TokenId> forced;
        const std::string text = ForcedBytes(automaton, s);
        if (!text.empty()) {
            forced = tokenizer.Encode(text, SpecialTokens::kAsText);
            int before_last = s;
            for (std::size_t i = 0; i + 1 < forced.size(); ++i) {
                before_last = Next(before_last, forced[i]);
            }
            const std::string_view last = tokenizer.TokenBytes(forced.back());
            const bool merges = std::ranges::any_of(transitions_[before_last], [&](const auto& transition) {
                const std::string_view bytes = tokenizer.TokenBytes(transition.first);
                return bytes.size() > last.size() && bytes.starts_with(last);
            });
            if (merges) {
                forced.pop_back();
            }
        } else if (transitions_[s].size() == 1) {
            forced = {transitions_[s].front().first};   // e.g. only the end token is left
        }
        forced_[s] = std::move(forced);
    }
}

// The state after `token` in `state`: kFinished after the end token, or
// ByteAutomaton::kDead if the grammar doesn't allow the token there.
int TokenGrammar::Next(int state, TokenId token) const {
    const auto& allowed = transitions_[state];
    const auto it = std::ranges::lower_bound(allowed, token, {}, &std::pair<TokenId, int>::first);
    return it != allowed.end() && it->first == token ? it->second : ByteAutomaton::kDead;
}

// `state`'s allowed tokens as a bitmask in GPU memory: bit id of word id / 32.
const std::uint32_t* TokenGrammar::DeviceMask(int state) const {
    return masks_.Data() + static_cast<std::size_t>(state) * mask_words_;
}

}  // namespace murmur
