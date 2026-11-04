#pragma once

// How weight matrices are stored, as the kernels see them.

#include <cstdint>

namespace murmur {

// A BF16 value stored as its raw 16 bits (the top half of an FP32).
using Bf16Bits = std::uint16_t;

enum class WeightFormat : std::uint8_t {
    kBf16,   // the checkpoint's 16-bit floats: 2 bytes per weight
    kFp8,    // 8-bit floats (E4M3) with one FP32 scale per output row: w = fp8 * scale
    kInt4,   // 4-bit integers with an FP16 {scale, min} per kInt4GroupSize weights of a row: w = q * scale + min
};

inline constexpr int kInt4GroupSize = 128;

// A weight matrix: out_features x in_features, row-major (PyTorch's layout),
// in one of the formats above. Rows are packed one after another; INT4 puts
// two weights in a byte, the even column in the low four bits. Non-owning:
// WeightMatrix owns the memory.
struct WeightView {
    WeightFormat format = WeightFormat::kBf16;
    int out_features = 0;
    int in_features = 0;
    const std::uint8_t* data = nullptr;
    const float* row_scales = nullptr;            // kFp8: out_features scales
    const std::uint32_t* group_ranges = nullptr;  // kInt4: out_features x in_features / kInt4GroupSize half2 {scale, min}
};

}  // namespace murmur
