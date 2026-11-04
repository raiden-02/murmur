#include "model/unicode.hpp"

#include <gtest/gtest.h>

#include <string>

namespace murmur::unicode {
namespace {

TEST(Utf8, DecodesEveryLength) {
    EXPECT_EQ(DecodeUtf8("A\xC3\xA9\xE6\x9D\xB1\xF0\x9F\x8D\x9C"), U"Aé東\U0001F35C");
}

TEST(Utf8, RoundTripsEveryLength) {
    const std::u32string text = U"Aé東\U0001F35C";
    EXPECT_EQ(DecodeUtf8(EncodeUtf8(text)), text);
}

TEST(Utf8, ReplacesEachMalformedByte) {
    EXPECT_EQ(DecodeUtf8("\xFF"), U"�");                    // never a lead byte
    EXPECT_EQ(DecodeUtf8("\x80" "a"), U"�a");               // stray continuation
    EXPECT_EQ(DecodeUtf8("\xC3"), U"�");                    // truncated at the end
    EXPECT_EQ(DecodeUtf8("\xC3" "a"), U"�a");               // missing continuation
    EXPECT_EQ(DecodeUtf8("\xC0\xAF"), U"��");          // overlong '/'
    EXPECT_EQ(DecodeUtf8("\xED\xA0\x80"), U"���");  // UTF-16 surrogate
    EXPECT_EQ(DecodeUtf8("\xF4\x90\x80\x80"), U"����");  // above U+10FFFF
}

TEST(Utf8, EncodesInvalidScalarsAsReplacement) {
    EXPECT_EQ(EncodeUtf8(std::u32string(1, char32_t{0xD800})), "\xEF\xBF\xBD");
    EXPECT_EQ(EncodeUtf8(std::u32string(1, char32_t{0x110000})), "\xEF\xBF\xBD");
}

TEST(CharacterClasses, MatchUnicodeCategories) {
    EXPECT_TRUE(IsLetter(U'a'));
    EXPECT_TRUE(IsLetter(U'é'));
    EXPECT_TRUE(IsLetter(U'東'));
    EXPECT_FALSE(IsLetter(U'1'));
    EXPECT_FALSE(IsLetter(U'\U0001F35C'));
    EXPECT_TRUE(IsNumber(U'7'));
    EXPECT_TRUE(IsNumber(U'½'));   // vulgar fraction one half (No)
    EXPECT_TRUE(IsNumber(U'٣'));   // Arabic-Indic digit three (Nd)
    EXPECT_FALSE(IsNumber(U'x'));
    EXPECT_TRUE(IsWhiteSpace(U' '));
    EXPECT_TRUE(IsWhiteSpace(U'\n'));
    EXPECT_TRUE(IsWhiteSpace(U'　'));   // ideographic space
    EXPECT_FALSE(IsWhiteSpace(U'​'));  // zero-width space is not White_Space
}

// Expected values come from Python's unicodedata.normalize("NFC", ...).
TEST(Nfc, ComposesAndReorders) {
    EXPECT_EQ(ToNfc(U"é"), U"é");
    EXPECT_EQ(ToNfc(U"ệ"), U"ệ");
    EXPECT_EQ(ToNfc(U"ệ"), U"ệ");
    EXPECT_EQ(ToNfc(U"q̣̇"), U"q̣̇");   // reordered, nothing to compose
    EXPECT_EQ(ToNfc(U"á́"), U"á́");    // second accent is blocked
}

TEST(Nfc, HandlesSingletonsExclusionsAndHangul) {
    EXPECT_EQ(ToNfc(U"Å"), U"Å");              // Angstrom sign -> A with ring
    EXPECT_EQ(ToNfc(U"क़"), U"क़");        // composition exclusion stays split
    EXPECT_EQ(ToNfc(U"각"), U"각");  // Hangul L + V + T
    EXPECT_EQ(ToNfc(U"각"), U"각");        // Hangul LV + T
    EXPECT_EQ(ToNfc(U"́e"), U"́e");            // leading mark has nothing to join
}

TEST(Nfc, LeavesAsciiAndComposedTextUnchanged) {
    EXPECT_EQ(ToNfc(U"plain ascii"), U"plain ascii");
    EXPECT_EQ(ToNfc(U"café 東京"), U"café 東京");
    EXPECT_EQ(ToNfc(U""), U"");
}

}  // namespace
}  // namespace murmur::unicode
