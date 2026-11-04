#include "model/tokenizer.hpp"

#include "model/pre_tokenizer.hpp"
#include "model/unicode.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>

namespace murmur {

namespace {

using Json = nlohmann::json;

// The pre-tokenizer this file implements (see pre_tokenizer.cpp). Loading a
// tokenizer.json with any other pattern would silently produce wrong ids.
constexpr std::string_view kExpectedSplitPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

[[noreturn]] void Fail(const std::string& message) {
    throw std::runtime_error("tokenizer.json: " + message);
}

Json ReadJson(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open " + path.string());
    }
    try {
        return Json::parse(file);
    } catch (const Json::exception& error) {
        throw std::runtime_error(path.string() + ": invalid JSON: " + error.what());
    }
}

// Byte-level BPE stores every byte as a printable character so token strings
// are valid text: bytes that are already printable map to themselves, the rest
// (control characters, space, ...) to U+0100 onwards. This is the inverse map,
// character -> byte, used to turn the vocabulary back into raw bytes.
std::unordered_map<char32_t, std::uint8_t> ByteLevelDecodeMap() {
    std::unordered_map<char32_t, std::uint8_t> map;
    char32_t next_substitute = 256;
    for (int b = 0; b < 256; ++b) {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        const char32_t character = printable ? static_cast<char32_t>(b) : next_substitute++;
        map.emplace(character, static_cast<std::uint8_t>(b));
    }
    return map;
}

std::string ByteLevelToBytes(std::string_view token, const std::unordered_map<char32_t, std::uint8_t>& decode_map) {
    std::string bytes;
    for (const char32_t c : unicode::DecodeUtf8(token)) {
        const auto it = decode_map.find(c);
        if (it == decode_map.end()) {
            Fail("vocabulary entry is not byte-level encoded: " + std::string(token));
        }
        bytes.push_back(static_cast<char>(it->second));
    }
    return bytes;
}

std::string StringAt(const Json& object, const char* key) {
    return object.at(key).get<std::string>();
}

void CheckSupportedFormat(const Json& json) {
    const Json& model = json.at("model");
    if (StringAt(model, "type") != "BPE" || model.value("byte_fallback", false) ||
        model.value("continuing_subword_prefix", std::string()) != "" ||
        model.value("end_of_word_suffix", std::string()) != "") {
        Fail("only plain byte-level BPE models are supported");
    }
    if (StringAt(json.at("normalizer"), "type") != "NFC") {
        Fail("expected an NFC normalizer");
    }
    const Json& steps = json.at("pre_tokenizer").at("pretokenizers");
    if (steps.size() != 2 || StringAt(steps[0].at("pattern"), "Regex") != kExpectedSplitPattern ||
        StringAt(steps[1], "type") != "ByteLevel" || steps[1].at("add_prefix_space").get<bool>()) {
        Fail("unsupported pre-tokenizer (expected the Qwen2 split pattern followed by ByteLevel)");
    }
}

std::uint64_t PairKey(TokenId left, TokenId right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) | static_cast<std::uint32_t>(right);
}

// Number of leading bytes of `bytes` that end on a UTF-8 character boundary:
// everything except an incomplete multi-byte sequence at the very end.
std::size_t CompleteUtf8Prefix(std::string_view bytes) {
    const std::size_t size = bytes.size();
    for (std::size_t back = 1; back <= std::min<std::size_t>(3, size); ++back) {
        const auto b = static_cast<std::uint8_t>(bytes[size - back]);
        if ((b & 0xC0) == 0x80) {
            continue;   // continuation byte: the sequence started earlier
        }
        const std::size_t expected = (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 1;
        return expected > back ? size - back : size;
    }
    return size;
}

// Replaces any malformed UTF-8 with U+FFFD so callers always get valid text.
std::string SanitizeUtf8(std::string_view bytes) {
    return unicode::EncodeUtf8(unicode::DecodeUtf8(bytes));
}

}  // namespace

// Loads a Hugging Face tokenizer.json for byte-level BPE with the Qwen2
// pre-tokenizer. The vocabulary is converted from its byte-level spelling to
// raw bytes once here, so encoding and decoding work on plain bytes. Every
// part of the file is validated: an unsupported format, a gap in the token
// ids or a merge whose result isn't in the vocabulary throws instead of
// producing wrong tokens later.
Tokenizer Tokenizer::FromFile(const std::filesystem::path& tokenizer_json) {
    const Json json = ReadJson(tokenizer_json);
    try {
        CheckSupportedFormat(json);
        Tokenizer tokenizer;
        const auto decode_map = ByteLevelDecodeMap();

        const Json& vocab = json.at("model").at("vocab");
        const Json& added = json.at("added_tokens");
        const std::size_t id_count = vocab.size() + added.size();
        if (id_count > static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
            Fail("too many tokens");
        }
        tokenizer.token_bytes_.resize(id_count);
        std::vector<bool> assigned(id_count, false);
        const auto assign = [&](TokenId id, std::string bytes) {
            if (id < 0 || static_cast<std::size_t>(id) >= id_count || assigned[id]) {
                Fail("token ids must be unique and contiguous from 0; bad id " + std::to_string(id));
            }
            assigned[id] = true;
            tokenizer.token_bytes_[id] = std::move(bytes);
        };

        std::unordered_map<std::string, TokenId> bytes_to_id;
        bytes_to_id.reserve(vocab.size());
        for (const auto& [spelling, id_json] : vocab.items()) {
            const auto id = id_json.get<TokenId>();
            std::string bytes = ByteLevelToBytes(spelling, decode_map);
            bytes_to_id.emplace(bytes, id);
            assign(id, std::move(bytes));
        }
        for (const Json& entry : added) {
            const auto id = entry.at("id").get<TokenId>();
            auto content = entry.at("content").get<std::string>();
            if (content.empty()) {
                Fail("empty added token");
            }
            tokenizer.added_tokens_.push_back({content, id});
            assign(id, std::move(content));
        }
        std::ranges::stable_sort(tokenizer.added_tokens_, std::ranges::greater{},
                                 [](const AddedToken& t) { return t.content.size(); });

        for (int b = 0; b < 256; ++b) {
            const auto it = bytes_to_id.find(std::string(1, static_cast<char>(b)));
            if (it == bytes_to_id.end()) {
                Fail("vocabulary is missing a single-byte token");
            }
            tokenizer.byte_tokens_[b] = it->second;
        }

        const Json& merges = json.at("model").at("merges");
        tokenizer.merges_.reserve(merges.size());
        for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
            const auto line = merges[rank].get<std::string>();
            const std::size_t space = line.find(' ');
            if (space == std::string::npos || line.find(' ', space + 1) != std::string::npos) {
                Fail("malformed merge \"" + line + "\"");
            }
            const std::string left = ByteLevelToBytes(std::string_view(line).substr(0, space), decode_map);
            const std::string right = ByteLevelToBytes(std::string_view(line).substr(space + 1), decode_map);
            const auto left_it = bytes_to_id.find(left);
            const auto right_it = bytes_to_id.find(right);
            const auto result_it = bytes_to_id.find(left + right);
            if (left_it == bytes_to_id.end() || right_it == bytes_to_id.end() || result_it == bytes_to_id.end()) {
                Fail("merge \"" + line + "\" refers to a token that isn't in the vocabulary");
            }
            // If a pair is listed twice, the first (lowest-rank) entry wins.
            tokenizer.merges_.try_emplace(PairKey(left_it->second, right_it->second), Merge{rank, result_it->second});
        }
        return tokenizer;
    } catch (const Json::exception& error) {
        Fail(std::string("unexpected structure: ") + error.what());
    }
}

// Text -> token ids. With SpecialTokens::kParse, any added token's exact
// spelling (longest first) becomes that token, as in Hugging Face; the text
// between them is encoded normally. Invalid UTF-8 is encoded as U+FFFD.
std::vector<TokenId> Tokenizer::Encode(std::string_view text, SpecialTokens special) const {
    std::vector<TokenId> ids;
    if (special == SpecialTokens::kAsText) {
        EncodeText(text, ids);
        return ids;
    }
    std::size_t segment_start = 0;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const auto match = std::ranges::find_if(
            added_tokens_, [&](const AddedToken& t) { return text.substr(pos).starts_with(t.content); });
        if (match == added_tokens_.end()) {
            ++pos;
            continue;
        }
        EncodeText(text.substr(segment_start, pos - segment_start), ids);
        ids.push_back(match->id);
        pos += match->content.size();
        segment_start = pos;
    }
    EncodeText(text.substr(segment_start), ids);
    return ids;
}

// Token ids -> the bytes they stand for, concatenated. The result is not
// guaranteed to be valid UTF-8 (a character can be split across a cut-off
// sequence); use StreamingDecoder for display.
std::string Tokenizer::Decode(std::span<const TokenId> ids) const {
    std::string text;
    for (const TokenId id : ids) {
        text += TokenBytes(id);
    }
    return text;
}

// The raw bytes token `id` stands for. Throws std::out_of_range for an id
// outside [0, VocabSize()).
std::string_view Tokenizer::TokenBytes(TokenId id) const {
    if (id < 0 || id >= VocabSize()) {
        throw std::out_of_range("Tokenizer: token id " + std::to_string(id) + " is not in the vocabulary");
    }
    return token_bytes_[id];
}

// Whether `id` is an added token such as "<|im_end|>" (chat structure, not text).
bool Tokenizer::IsSpecial(TokenId id) const {
    return std::ranges::any_of(added_tokens_, [&](const AddedToken& token) { return token.id == id; });
}

// The id of an added token such as "<|im_end|>". Throws std::invalid_argument
// if there is no such token.
TokenId Tokenizer::SpecialTokenId(std::string_view content) const {
    const auto it = std::ranges::find(added_tokens_, content, &AddedToken::content);
    if (it == added_tokens_.end()) {
        throw std::invalid_argument("Tokenizer: no special token " + std::string(content));
    }
    return it->id;
}

// Number of token ids, all valid: [0, VocabSize()).
std::int32_t Tokenizer::VocabSize() const {
    return static_cast<std::int32_t>(token_bytes_.size());
}

// Ordinary text (no special tokens): NFC-normalize, split into pieces with
// the pre-tokenizer, then run BPE on each piece's UTF-8 bytes.
void Tokenizer::EncodeText(std::string_view text, std::vector<TokenId>& out) const {
    if (text.empty()) {
        return;
    }
    const std::u32string normalized = unicode::ToNfc(unicode::DecodeUtf8(text));
    for (const std::u32string_view piece : PreTokenize(normalized)) {
        EncodePiece(unicode::EncodeUtf8(piece), out);
    }
}

// Byte-pair encoding of one piece: start with one token per byte, then
// repeatedly merge the adjacent pair with the lowest merge rank (leftmost
// first on ties) until no listed pair is left.
//
// Symbols form a linked list and candidate merges sit in a min-heap, so a
// piece of n bytes costs O(n log n). A heap entry goes stale when either of
// its symbols has since merged with something else; it is then skipped.
// Token ids only ever grow into longer tokens, so an unchanged id on both
// sides proves the pair is still current.
void Tokenizer::EncodePiece(std::string_view bytes, std::vector<TokenId>& out) const {
    constexpr int kNone = -1;
    struct Symbol {
        TokenId id;
        int prev;
        int next;
    };
    struct Candidate {
        std::uint32_t rank;
        int left;   // index of the left symbol; its position breaks rank ties
        TokenId left_id;
        TokenId right_id;
        TokenId result;
        bool operator>(const Candidate& other) const {
            return std::pair{rank, left} > std::pair{other.rank, other.left};
        }
    };

    const int count = static_cast<int>(bytes.size());
    std::vector<Symbol> symbols(count);
    for (int i = 0; i < count; ++i) {
        symbols[i] = {byte_tokens_[static_cast<std::uint8_t>(bytes[i])], i - 1, i + 1 < count ? i + 1 : kNone};
    }

    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> heap;
    const auto push_pair = [&](int left) {
        if (left == kNone || symbols[left].next == kNone) {
            return;
        }
        const TokenId left_id = symbols[left].id;
        const TokenId right_id = symbols[symbols[left].next].id;
        if (const Merge* merge = FindMerge(left_id, right_id)) {
            heap.push({merge->rank, left, left_id, right_id, merge->result});
        }
    };
    for (int i = 0; i + 1 < count; ++i) {
        push_pair(i);
    }

    while (!heap.empty()) {
        const Candidate candidate = heap.top();
        heap.pop();
        Symbol& left = symbols[candidate.left];
        if (left.id != candidate.left_id || left.next == kNone || symbols[left.next].id != candidate.right_id) {
            continue;   // stale
        }
        Symbol& right = symbols[left.next];
        left.id = candidate.result;
        left.next = right.next;
        if (right.next != kNone) {
            symbols[right.next].prev = candidate.left;
        }
        right.id = kNone;
        push_pair(left.prev);
        push_pair(candidate.left);
    }

    for (int i = count > 0 ? 0 : kNone; i != kNone; i = symbols[i].next) {
        out.push_back(symbols[i].id);
    }
}

const Tokenizer::Merge* Tokenizer::FindMerge(TokenId left, TokenId right) const {
    const auto it = merges_.find(PairKey(left, right));
    return it == merges_.end() ? nullptr : &it->second;
}

StreamingDecoder::StreamingDecoder(const Tokenizer& tokenizer) : tokenizer_(&tokenizer) {}

// Adds one token and returns the text it completes: always valid UTF-8, and
// possibly empty while a multi-byte character is still waiting for its rest.
std::string StreamingDecoder::Push(TokenId id) {
    pending_ += tokenizer_->TokenBytes(id);
    const std::size_t complete = CompleteUtf8Prefix(pending_);
    std::string text = SanitizeUtf8(std::string_view(pending_).substr(0, complete));
    pending_.erase(0, complete);
    return text;
}

// Returns whatever is still pending (an unfinished character becomes U+FFFD)
// and resets the decoder.
std::string StreamingDecoder::Flush() {
    std::string text = SanitizeUtf8(pending_);
    pending_.clear();
    return text;
}

}  // namespace murmur
