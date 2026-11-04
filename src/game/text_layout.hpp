#pragma once

// Fonts and word-wrapping for dialogue text (UTF-8).

#include "raylib.h"

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace murmur::game {

// A TTF font with Latin and common punctuation glyphs, loaded from the
// system; falls back to raylib's built-in ASCII font. Owns the GPU texture.
class UiFont {
public:
    explicit UiFont(int base_size);
    ~UiFont();

    UiFont(const UiFont&) = delete;
    UiFont& operator=(const UiFont&) = delete;

    const Font& Get() const { return font_; }

private:
    Font font_{};
    bool owned_ = false;
};

// How wide a piece of text is drawn, in pixels.
using TextMeasure = std::function<float(std::string_view)>;

// Lines are views into `text`: keep it alive while they are used.
std::vector<std::string_view> WrapText(std::string_view text, float max_width, const TextMeasure& width);
std::vector<std::string_view> WrapText(const Font& font, std::string_view text, float font_size, float max_width);

void PopLastCodepoint(std::string& utf8);

}  // namespace murmur::game
