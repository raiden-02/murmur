#pragma once

#include <cstddef>
#include <stdexcept>

namespace murmur {

// Blocks needed to cover `count` items with `per_block` items per block (ceil).
constexpr int CeilDiv(std::size_t count, int per_block) {
    return static_cast<int>((count + per_block - 1) / per_block);
}

// Launchers validate their arguments on the host before touching the GPU,
// so a bad shape fails loudly instead of corrupting memory.
inline void RequireArgument(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

}  // namespace murmur
