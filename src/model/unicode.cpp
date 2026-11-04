#include "model/unicode.hpp"

#include "model/unicode_tables.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
#include <utility>

namespace murmur::unicode {

namespace {

constexpr char32_t kMaxCodepoint = 0x10FFFF;
constexpr char32_t kSurrogateFirst = 0xD800;
constexpr char32_t kSurrogateLast = 0xDFFF;

// Code points below this are never changed by NFC (checked by the generator).
constexpr char32_t kNfcFastPathLimit = 0x300;

// Hangul syllables are composed of leading consonant (L), vowel (V) and an
// optional trailing consonant (T); Unicode defines their (de)composition
// arithmetically (Unicode standard, section 3.12).
constexpr char32_t kHangulSBase = 0xAC00;
constexpr char32_t kHangulLBase = 0x1100;
constexpr char32_t kHangulVBase = 0x1161;
constexpr char32_t kHangulTBase = 0x11A7;
constexpr char32_t kHangulLCount = 19;
constexpr char32_t kHangulVCount = 21;
constexpr char32_t kHangulTCount = 28;
constexpr char32_t kHangulNCount = kHangulVCount * kHangulTCount;
constexpr char32_t kHangulSCount = kHangulLCount * kHangulNCount;

bool InRanges(std::span<const tables::CodepointRange> ranges, char32_t codepoint) {
    const auto it = std::upper_bound(ranges.begin(), ranges.end(), codepoint,
                                     [](char32_t cp, const tables::CodepointRange& r) { return cp < r.first; });
    return it != ranges.begin() && codepoint <= std::prev(it)->last;
}

std::uint8_t CombiningClass(char32_t codepoint) {
    const auto ranges = tables::CombiningClassRanges();
    const auto it = std::upper_bound(ranges.begin(), ranges.end(), codepoint,
                                     [](char32_t cp, const tables::CombiningClassRange& r) { return cp < r.first; });
    if (it == ranges.begin() || codepoint > std::prev(it)->last) {
        return 0;
    }
    return std::prev(it)->combining_class;
}

// Appends the full canonical decomposition of `codepoint` (applied recursively).
void AppendDecomposition(char32_t codepoint, std::u32string& out) {
    if (codepoint >= kHangulSBase && codepoint < kHangulSBase + kHangulSCount) {
        const char32_t index = codepoint - kHangulSBase;
        out.push_back(kHangulLBase + index / kHangulNCount);
        out.push_back(kHangulVBase + (index % kHangulNCount) / kHangulTCount);
        if (const char32_t t = index % kHangulTCount; t != 0) {
            out.push_back(kHangulTBase + t);
        }
        return;
    }
    const auto entries = tables::Decompositions();
    const auto it = std::lower_bound(entries.begin(), entries.end(), codepoint,
                                     [](const tables::Decomposition& d, char32_t cp) { return d.codepoint < cp; });
    if (it == entries.end() || it->codepoint != codepoint) {
        out.push_back(codepoint);
        return;
    }
    for (const char32_t part : tables::DecompositionData().subspan(it->offset, it->length)) {
        AppendDecomposition(part, out);
    }
}

// The primary composite of the pair (first, second), if there is one.
std::optional<char32_t> Compose(char32_t first, char32_t second) {
    if (first >= kHangulLBase && first < kHangulLBase + kHangulLCount &&
        second >= kHangulVBase && second < kHangulVBase + kHangulVCount) {
        return kHangulSBase + ((first - kHangulLBase) * kHangulVCount + (second - kHangulVBase)) * kHangulTCount;
    }
    if (first >= kHangulSBase && first < kHangulSBase + kHangulSCount && (first - kHangulSBase) % kHangulTCount == 0 &&
        second > kHangulTBase && second < kHangulTBase + kHangulTCount) {
        return first + (second - kHangulTBase);
    }
    const auto pairs = tables::Compositions();
    const auto it = std::lower_bound(pairs.begin(), pairs.end(), std::pair{first, second},
                                     [](const tables::Composition& c, const std::pair<char32_t, char32_t>& key) {
                                         return std::pair{c.first, c.second} < key;
                                     });
    if (it == pairs.end() || it->first != first || it->second != second) {
        return std::nullopt;
    }
    return it->composite;
}

// Sorts each run of combining marks by combining class, keeping equal classes
// in their original order (canonical ordering).
void CanonicalOrder(std::u32string& text) {
    auto run_start = text.begin();
    while (run_start != text.end()) {
        run_start = std::find_if(run_start, text.end(), [](char32_t cp) { return CombiningClass(cp) != 0; });
        const auto run_end = std::find_if(run_start, text.end(), [](char32_t cp) { return CombiningClass(cp) == 0; });
        std::stable_sort(run_start, run_end,
                         [](char32_t a, char32_t b) { return CombiningClass(a) < CombiningClass(b); });
        run_start = run_end;
    }
}

// Canonical composition of decomposed, canonically ordered text, in place
// (the algorithm of UAX #15, section 1.3).
void CanonicalCompose(std::u32string& text) {
    if (text.empty()) {
        return;
    }
    std::size_t starter = 0;
    // A leading combining mark has no starter to compose with: 256 blocks it.
    int last_class = CombiningClass(text[0]) == 0 ? 0 : 256;
    std::size_t write = 1;
    for (std::size_t read = 1; read < text.size(); ++read) {
        const char32_t cp = text[read];
        const int cp_class = CombiningClass(cp);
        // `cp` may join the starter only if nothing in between has an equal or higher class.
        const std::optional<char32_t> composite = Compose(text[starter], cp);
        if (composite && (last_class < cp_class || last_class == 0)) {
            text[starter] = *composite;
            continue;
        }
        if (cp_class == 0) {
            starter = write;
        }
        last_class = cp_class;
        text[write++] = cp;
    }
    text.resize(write);
}

struct DecodedCodepoint {
    char32_t codepoint;
    std::size_t length;   // bytes consumed
};

// Decodes the well-formed UTF-8 sequence at the start of `bytes`, or returns
// nullopt if there isn't one.
std::optional<DecodedCodepoint> DecodeOne(std::string_view bytes) {
    const auto byte_at = [&](std::size_t i) { return static_cast<std::uint8_t>(bytes[i]); };
    const std::uint8_t lead = byte_at(0);

    std::size_t length = 0;
    char32_t codepoint = 0;
    char32_t min_value = 0;   // smaller values in this length are overlong encodings
    if (lead < 0x80) {
        return DecodedCodepoint{lead, 1};
    } else if ((lead & 0xE0) == 0xC0) {
        length = 2;
        codepoint = lead & 0x1F;
        min_value = 0x80;
    } else if ((lead & 0xF0) == 0xE0) {
        length = 3;
        codepoint = lead & 0x0F;
        min_value = 0x800;
    } else if ((lead & 0xF8) == 0xF0) {
        length = 4;
        codepoint = lead & 0x07;
        min_value = 0x10000;
    } else {
        return std::nullopt;
    }
    if (bytes.size() < length) {
        return std::nullopt;
    }
    for (std::size_t k = 1; k < length; ++k) {
        if ((byte_at(k) & 0xC0) != 0x80) {
            return std::nullopt;
        }
        codepoint = (codepoint << 6) | (byte_at(k) & 0x3F);
    }
    const bool is_surrogate = codepoint >= kSurrogateFirst && codepoint <= kSurrogateLast;
    if (codepoint < min_value || codepoint > kMaxCodepoint || is_surrogate) {
        return std::nullopt;
    }
    return DecodedCodepoint{codepoint, length};
}

}  // namespace

// Decodes UTF-8 into code points. Each byte that doesn't start a well-formed
// sequence (bad lead byte, missing or bad continuation byte, overlong form,
// surrogate, or a value above U+10FFFF) becomes one U+FFFD, and decoding
// resumes at the next byte, so untrusted input can never cause an error.
std::u32string DecodeUtf8(std::string_view text) {
    std::u32string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (const std::optional<DecodedCodepoint> decoded = DecodeOne(text.substr(i))) {
            out.push_back(decoded->codepoint);
            i += decoded->length;
        } else {
            out.push_back(kReplacementCharacter);
            ++i;
        }
    }
    return out;
}

// Appends `codepoint` as UTF-8. Surrogates and values above U+10FFFF are not
// valid scalar values and are written as U+FFFD.
void AppendUtf8(char32_t codepoint, std::string& out) {
    if (codepoint > kMaxCodepoint || (codepoint >= kSurrogateFirst && codepoint <= kSurrogateLast)) {
        codepoint = kReplacementCharacter;
    }
    if (codepoint < 0x80) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

std::string EncodeUtf8(std::u32string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char32_t cp : text) {
        AppendUtf8(cp, out);
    }
    return out;
}

// General category L (Lu, Ll, Lt, Lm, Lo): the regex class \p{L}.
bool IsLetter(char32_t codepoint) {
    if (codepoint < 0x80) {
        return (codepoint | 0x20) >= U'a' && (codepoint | 0x20) <= U'z';
    }
    return InRanges(tables::LetterRanges(), codepoint);
}

// General category N (Nd, Nl, No): the regex class \p{N}.
bool IsNumber(char32_t codepoint) {
    if (codepoint < 0x80) {
        return codepoint >= U'0' && codepoint <= U'9';
    }
    return InRanges(tables::NumberRanges(), codepoint);
}

// The Unicode White_Space property: the regex class \s.
bool IsWhiteSpace(char32_t codepoint) {
    return InRanges(tables::WhiteSpaceRanges(), codepoint);
}

// Unicode Normalization Form C: canonical decomposition, canonical ordering,
// then canonical composition. Text made only of code points below U+0300 is
// already in NFC and is returned unchanged without any table lookups.
std::u32string ToNfc(std::u32string_view text) {
    if (std::all_of(text.begin(), text.end(), [](char32_t cp) { return cp < kNfcFastPathLimit; })) {
        return std::u32string(text);
    }
    std::u32string result;
    result.reserve(text.size());
    for (const char32_t cp : text) {
        AppendDecomposition(cp, result);
    }
    CanonicalOrder(result);
    CanonicalCompose(result);
    return result;
}

}  // namespace murmur::unicode
