#include "game/text_layout.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

namespace murmur::game {

namespace {

constexpr float kLetterSpacing = 1.0f;

// Glyphs to bake into the font atlas: ASCII, Latin-1 and Latin Extended-A
// (accented letters), and the typographic punctuation models like to use.
std::vector<int> DialogueCodepoints() {
    std::vector<int> codepoints;
    const auto add_range = [&](int first, int last) {
        for (int c = first; c <= last; ++c) {
            codepoints.push_back(c);
        }
    };
    add_range(0x20, 0x7E);
    add_range(0xA0, 0x17F);
    add_range(0x2010, 0x2027);   // dashes, curly quotes, bullet, ellipsis
    add_range(0x2030, 0x203A);
    codepoints.push_back(0x20AC);   // euro sign
    codepoints.push_back(0x2122);   // trade mark sign
    return codepoints;
}

std::filesystem::path SystemFontPath() {
    char* windir = nullptr;
    std::size_t length = 0;
    std::filesystem::path path = "C:/Windows";
    if (_dupenv_s(&windir, &length, "WINDIR") == 0 && windir != nullptr) {
        path = windir;
        std::free(windir);
    }
    return path / "Fonts" / "segoeui.ttf";
}

// raylib measures null-terminated strings, so the piece is copied first.
float Width(const Font& font, std::string_view text, float font_size) {
    return MeasureTextEx(font, std::string(text).c_str(), font_size, kLetterSpacing).x;
}

// Bytes in the UTF-8 character that starts at text[at]: 1 for ASCII and for
// a malformed byte, so a broken sequence is stepped over a byte at a time.
std::size_t CharacterBytes(std::string_view text, std::size_t at) {
    const auto lead = static_cast<unsigned char>(text[at]);
    const std::size_t length = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    return std::min(length, text.size() - at);
}

}  // namespace

// Loads Segoe UI at `base_size` pixels (sharpest when drawn at that size).
// If the system font is missing, raylib's default font is used instead and
// non-ASCII characters show as '?'.
UiFont::UiFont(int base_size) {
    const std::filesystem::path path = SystemFontPath();
    if (std::filesystem::exists(path)) {
        std::vector<int> codepoints = DialogueCodepoints();
        font_ = LoadFontEx(path.string().c_str(), base_size, codepoints.data(), static_cast<int>(codepoints.size()));
        owned_ = IsFontValid(font_);
    }
    if (owned_) {
        SetTextureFilter(font_.texture, TEXTURE_FILTER_BILINEAR);
    } else {
        font_ = GetFontDefault();
    }
}

UiFont::~UiFont() {
    if (owned_) {
        UnloadFont(font_);
    }
}

// Breaks `text` into lines no wider than `max_width` as measured by `width`:
// at spaces where possible, inside a word only when the word alone is too
// wide (between characters, never inside one). Newlines in the text always
// start a new line. Each line is a piece of `text` itself, spaces between
// its words included, so a caller can find any line's bytes in the text
// (line.data() - text.data()) to colour them byte by byte.
std::vector<std::string_view> WrapText(std::string_view text, float max_width, const TextMeasure& width) {
    std::vector<std::string_view> lines;
    std::optional<std::size_t> line_begin;   // the open line's first byte; none until a word lands
    std::size_t line_end = 0;
    const auto close_line = [&](std::size_t at) {   // an empty line sits at `at`
        const std::size_t begin = line_begin.value_or(at);
        lines.push_back(text.substr(begin, line_begin ? line_end - begin : 0));
        line_begin.reset();
    };
    std::size_t at = 0;
    while (at < text.size()) {
        if (text[at] == ' ') {
            ++at;
            continue;
        }
        if (text[at] == '\n') {
            close_line(at);
            ++at;
            continue;
        }
        const std::size_t word_end = std::min(text.find_first_of(" \n", at), text.size());
        if (line_begin && width(text.substr(*line_begin, word_end - *line_begin)) <= max_width) {
            line_end = word_end;   // the word, and the spaces before it, join the line
        } else {
            if (line_begin) {
                close_line(at);
            }
            // The word starts a line; if it is wider than a whole line, its
            // first pieces fill lines of their own.
            std::size_t piece = at;
            for (std::size_t next = at; next < word_end;) {
                const std::size_t after = next + CharacterBytes(text, next);
                if (next > piece && width(text.substr(piece, after - piece)) > max_width) {
                    lines.push_back(text.substr(piece, next - piece));
                    piece = next;
                    continue;
                }
                next = after;
            }
            line_begin = piece;
            line_end = word_end;
        }
        at = word_end;
    }
    if (line_begin) {
        close_line(text.size());
    }
    return lines;
}

// WrapText for text drawn with `font` at `font_size`.
std::vector<std::string_view> WrapText(const Font& font, std::string_view text, float font_size, float max_width) {
    return WrapText(text, max_width, [&](std::string_view piece) { return Width(font, piece, font_size); });
}

// Removes the last character (not just the last byte) from a UTF-8 string,
// for backspace in a text box.
void PopLastCodepoint(std::string& utf8) {
    while (!utf8.empty()) {
        const auto byte = static_cast<unsigned char>(utf8.back());
        utf8.pop_back();
        if ((byte & 0xC0) != 0x80) {
            break;   // removed the lead byte: the character is gone
        }
    }
}

}  // namespace murmur::game
