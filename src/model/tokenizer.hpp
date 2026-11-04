#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace murmur {

using TokenId = std::int32_t;

// Whether text that spells a special token (e.g. "<|im_end|>") becomes that
// token. Text from players must use kAsText, or it could forge chat structure.
enum class SpecialTokens {
    kParse,
    kAsText,
};

class Tokenizer {
public:
    static Tokenizer FromFile(const std::filesystem::path& tokenizer_json);

    std::vector<TokenId> Encode(std::string_view text, SpecialTokens special) const;
    std::string Decode(std::span<const TokenId> ids) const;

    std::string_view TokenBytes(TokenId id) const;
    TokenId SpecialTokenId(std::string_view content) const;
    bool IsSpecial(TokenId id) const;
    std::int32_t VocabSize() const;

private:
    struct Merge {
        std::uint32_t rank;
        TokenId result;
    };
    struct AddedToken {
        std::string content;
        TokenId id;
    };

    Tokenizer() = default;

    void EncodeText(std::string_view text, std::vector<TokenId>& out) const;
    void EncodePiece(std::string_view bytes, std::vector<TokenId>& out) const;
    const Merge* FindMerge(TokenId left, TokenId right) const;

    std::vector<std::string> token_bytes_;               // id -> the bytes it stands for
    std::array<TokenId, 256> byte_tokens_{};             // single byte -> its token
    std::unordered_map<std::uint64_t, Merge> merges_;    // (left, right) -> merge
    std::vector<AddedToken> added_tokens_;               // longest content first
};

// Turns a stream of token ids into text without ever splitting a UTF-8
// character: an emoji spread over two tokens comes out whole, once both arrive.
class StreamingDecoder {
public:
    explicit StreamingDecoder(const Tokenizer& tokenizer);

    std::string Push(TokenId id);
    std::string Flush();

private:
    const Tokenizer* tokenizer_;
    std::string pending_;
};

}  // namespace murmur
