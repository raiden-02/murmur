#pragma once

#include <string>
#include <string_view>

namespace murmur::unicode {

inline constexpr char32_t kReplacementCharacter = U'�';

std::u32string DecodeUtf8(std::string_view text);
void AppendUtf8(char32_t codepoint, std::string& out);
std::string EncodeUtf8(std::u32string_view text);

bool IsLetter(char32_t codepoint);
bool IsNumber(char32_t codepoint);
bool IsWhiteSpace(char32_t codepoint);

std::u32string ToNfc(std::u32string_view text);

}  // namespace murmur::unicode
