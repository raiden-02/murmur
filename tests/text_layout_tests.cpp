#include "game/text_layout.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

namespace murmur::game {
namespace {

// Every character 10 pixels wide, so a width of 100 holds 10 characters.
float Monospace(std::string_view text) {
    float width = 0.0f;
    for (const char c : text) {
        width += (static_cast<unsigned char>(c) & 0xC0) == 0x80 ? 0.0f : 10.0f;   // continuation bytes add nothing
    }
    return width;
}

std::vector<std::string> Wrap(std::string_view text, float max_width) {
    std::vector<std::string> lines;
    for (const std::string_view line : WrapText(text, max_width, Monospace)) {
        lines.emplace_back(line);
    }
    return lines;
}

using Lines = std::vector<std::string>;

TEST(WrapText, BreaksAtSpacesWherePossible) {
    EXPECT_EQ(Wrap("the quick brown fox", 100), (Lines{"the quick", "brown fox"}));
    EXPECT_EQ(Wrap("  leading and trailing  ", 100), (Lines{"leading", "and", "trailing"}));
    EXPECT_EQ(Wrap("", 100), Lines{});
}

TEST(WrapText, StartsANewLineAtEveryNewline) {
    EXPECT_EQ(Wrap("one\n\ntwo", 100), (Lines{"one", "", "two"}));
    EXPECT_EQ(Wrap("one\n", 100), (Lines{"one"}));
}

// A word wider than a line is split between characters, never inside one.
TEST(WrapText, SplitsAWordTooWideForALine) {
    EXPECT_EQ(Wrap("abcdefghijklmno", 50), (Lines{"abcde", "fghij", "klmno"}));
    EXPECT_EQ(Wrap("\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9", 30),   // five é, two bytes each
              (Lines{"\xC3\xA9\xC3\xA9\xC3\xA9", "\xC3\xA9\xC3\xA9"}));
    EXPECT_EQ(Wrap("ab abcdefgh", 50), (Lines{"ab", "abcde", "fgh"}));
}

// The speech bubble colours a reply byte by byte (the draft model's guesses
// in teal), so every line must be the very bytes of the reply it came from,
// spaces inside it included, at the place they are.
TEST(WrapText, EveryLineIsAPieceOfTheText) {
    const std::string text = " Fresh noodles,  two  for one.\nCome  back  later";
    const std::vector<std::string_view> lines = WrapText(text, 100, Monospace);
    ASSERT_EQ(lines.size(), 6u);
    EXPECT_EQ(lines[2], "two  for");
    EXPECT_EQ(lines[4], "Come  back");
    std::size_t previous_end = 0;
    for (const std::string_view line : lines) {
        ASSERT_GE(line.data(), text.data());
        const auto offset = static_cast<std::size_t>(line.data() - text.data());
        ASSERT_LE(offset + line.size(), text.size());
        EXPECT_EQ(text.substr(offset, line.size()), line);
        EXPECT_GE(offset, previous_end) << "lines come in order and don't overlap";
        previous_end = offset + line.size();
    }
}

}  // namespace
}  // namespace murmur::game
