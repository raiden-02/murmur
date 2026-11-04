#include "model/pre_tokenizer.hpp"

#include "model/unicode.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace murmur {
namespace {

std::vector<std::string> Pieces(const std::string& text) {
    const std::u32string decoded = unicode::DecodeUtf8(text);
    std::vector<std::string> pieces;
    for (const std::u32string_view piece : PreTokenize(decoded)) {
        pieces.push_back(unicode::EncodeUtf8(piece));
    }
    return pieces;
}

using Expected = std::vector<std::string>;

// Expected splits come from the Hugging Face pre-tokenizer for this model.
TEST(PreTokenize, AttachesOneLeadingSpaceToWords) {
    EXPECT_EQ(Pieces("Hello world"), (Expected{"Hello", " world"}));
    EXPECT_EQ(Pieces("  hi there"), (Expected{" ", " hi", " there"}));
    EXPECT_EQ(Pieces("x  y"), (Expected{"x", " ", " y"}));
    EXPECT_EQ(Pieces("tab\there"), (Expected{"tab", "\there"}));
}

TEST(PreTokenize, GroupsWhitespaceAndLineBreaks) {
    EXPECT_EQ(Pieces("a  \n\n b"), (Expected{"a", "  \n\n", " b"}));
    EXPECT_EQ(Pieces("a\r\nb"), (Expected{"a", "\r\n", "b"}));
    EXPECT_EQ(Pieces("end   "), (Expected{"end", "   "}));
}

TEST(PreTokenize, SplitsContractionsDigitsAndSymbols) {
    EXPECT_EQ(Pieces("I'M here"), (Expected{"I", "'M", " here"}));
    EXPECT_EQ(Pieces("12345"), (Expected{"1", "2", "3", "4", "5"}));
    EXPECT_EQ(Pieces("!!! ok"), (Expected{"!!!", " ok"}));
}

TEST(PreTokenize, CoversTheWholeText) {
    const std::string text = "Mixed: caf\xC3\xA9, 42 \xF0\x9F\x8D\x9C\n\n  'tis   done!";
    std::string joined;
    for (const std::string& piece : Pieces(text)) {
        EXPECT_FALSE(piece.empty());
        joined += piece;
    }
    EXPECT_EQ(joined, text);
}

TEST(PreTokenize, EmptyTextHasNoPieces) {
    EXPECT_TRUE(Pieces("").empty());
}

}  // namespace
}  // namespace murmur
