#include "model/pre_tokenizer.hpp"

#include "model/unicode.hpp"

#include <cstddef>

namespace murmur {

namespace {

using unicode::IsLetter;
using unicode::IsNumber;
using unicode::IsWhiteSpace;

bool IsLineBreak(char32_t c) {
    return c == U'\r' || c == U'\n';
}

// [^\s\p{L}\p{N}]: punctuation, symbols, emoji and other non-word characters.
bool IsOther(char32_t c) {
    return !IsWhiteSpace(c) && !IsLetter(c) && !IsNumber(c);
}

// Case-insensitive ASCII letter match. U+017F (long s) case-folds to 's'.
bool MatchesFolded(char32_t c, char lower) {
    if (lower == 's' && c == U'ſ') {
        return true;
    }
    return c < 0x80 && static_cast<char>(c | 0x20) == lower;
}

// Finds the regex's match at a fixed start position. Each alternative of
// the pattern is one member function returning the match length, or 0 when
// it doesn't match there; Match() tries them in pattern order, which is the
// leftmost-first rule the original regex engine uses.
class PieceMatcher {
public:
    explicit PieceMatcher(std::u32string_view text) : text_(text) {}

    std::size_t Match(std::size_t pos) const {
        for (const auto alternative : {&PieceMatcher::Contraction, &PieceMatcher::Word, &PieceMatcher::Digit,
                                       &PieceMatcher::Symbols, &PieceMatcher::LineBreaks,
                                       &PieceMatcher::SpacesBeforeSpace, &PieceMatcher::Spaces}) {
            if (const std::size_t length = (this->*alternative)(pos); length > 0) {
                return length;
            }
        }
        return 1;   // unreachable: every character starts some alternative
    }

private:
    char32_t At(std::size_t pos) const { return pos < text_.size() ? text_[pos] : U'\0'; }

    std::size_t RunLength(std::size_t pos, bool (*predicate)(char32_t)) const {
        std::size_t end = pos;
        while (end < text_.size() && predicate(text_[end])) {
            ++end;
        }
        return end - pos;
    }

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    std::size_t Contraction(std::size_t pos) const {
        if (At(pos) != U'\'') {
            return 0;
        }
        for (const std::string_view suffix : {"s", "t", "re", "ve", "m", "ll", "d"}) {
            bool matches = true;
            for (std::size_t k = 0; k < suffix.size() && matches; ++k) {
                matches = MatchesFolded(At(pos + 1 + k), suffix[k]);
            }
            if (matches) {
                return 1 + suffix.size();
            }
        }
        return 0;
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+ : a word, optionally led by one space or symbol.
    std::size_t Word(std::size_t pos) const {
        const char32_t first = At(pos);
        if (pos < text_.size() && !IsLineBreak(first) && !IsLetter(first) && !IsNumber(first)) {
            if (const std::size_t letters = RunLength(pos + 1, IsLetter); letters > 0) {
                return 1 + letters;
            }
        }
        return RunLength(pos, IsLetter);
    }

    // \p{N} : numbers are split into single digits.
    std::size_t Digit(std::size_t pos) const {
        return pos < text_.size() && IsNumber(text_[pos]) ? 1 : 0;
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n]* : a symbol run, optionally led by one space,
    // swallowing the line breaks right after it.
    std::size_t Symbols(std::size_t pos) const {
        const std::size_t start = At(pos) == U' ' ? pos + 1 : pos;
        const std::size_t symbols = RunLength(start, IsOther);
        if (symbols == 0) {
            return 0;
        }
        const std::size_t end = start + symbols;
        return end + RunLength(end, IsLineBreak) - pos;
    }

    // \s*[\r\n]+ : whitespace up to and including its last line break.
    std::size_t LineBreaks(std::size_t pos) const {
        const std::size_t run = RunLength(pos, IsWhiteSpace);
        for (std::size_t end = pos + run; end > pos; --end) {
            if (IsLineBreak(text_[end - 1])) {
                return end - pos;
            }
        }
        return 0;
    }

    // \s+(?!\S) : whitespace, minus its last character when a non-space
    // follows, so that character can lead the next word (" hello").
    std::size_t SpacesBeforeSpace(std::size_t pos) const {
        const std::size_t run = RunLength(pos, IsWhiteSpace);
        if (run == 0) {
            return 0;
        }
        if (pos + run == text_.size()) {
            return run;
        }
        return run >= 2 ? run - 1 : 0;
    }

    // \s+
    std::size_t Spaces(std::size_t pos) const {
        return RunLength(pos, IsWhiteSpace);
    }

    std::u32string_view text_;
};

}  // namespace

// Splits text into the pieces that BPE runs on, matching the Qwen2
// pre-tokenizer regex exactly:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// The pattern matches every character, so the pieces cover the whole text
// with no gaps. A hand-written matcher replaces a regex engine because the
// standard library's <regex> has no Unicode classes, and this one runs in
// linear time on any input.
std::vector<std::u32string_view> PreTokenize(std::u32string_view text) {
    std::vector<std::u32string_view> pieces;
    const PieceMatcher matcher(text);
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t length = matcher.Match(pos);
        pieces.push_back(text.substr(pos, length));
        pos += length;
    }
    return pieces;
}

}  // namespace murmur
