#pragma once

#include <string_view>
#include <vector>

namespace murmur {

std::vector<std::u32string_view> PreTokenize(std::u32string_view text);

}  // namespace murmur
