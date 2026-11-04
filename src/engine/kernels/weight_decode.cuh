#pragma once

// Device-side unpacking of the weight formats (see weight_view.cuh), shared
// by the kernels that read quantized weights.

#include "engine/kernels/weight_view.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>

namespace murmur {

__device__ __forceinline__ float Bf16ToFloat(Bf16Bits bits) {
    return __uint_as_float(static_cast<unsigned>(bits) << 16);
}

// Two E4M3 values (the lower address in the low byte) as an FP16 pair.
// Every E4M3 value is exactly representable in FP16, and Ada converts a pair
// in one instruction.
__device__ __forceinline__ __half2 Fp8PairToHalf2(std::uint16_t pair) {
    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
    return *reinterpret_cast<const __half2*>(&raw);
}

__device__ __forceinline__ float2 Fp8PairToFloat2(std::uint16_t pair) {
    return __half22float2(Fp8PairToHalf2(pair));
}

// Nibble i (bits 4i..4i+3) of `word` as a float, exactly: OR-ing it into the
// mantissa of 2^23 gives 2^23 + q, and subtracting 2^23 leaves q. Two cheap
// integer/float operations instead of an int-to-float conversion.
__device__ __forceinline__ float NibbleToFloat(unsigned word, int i) {
    return __uint_as_float(0x4B000000u | ((word >> (4 * i)) & 0xFu)) - 8388608.0f;
}

// The two nibbles of a byte (low one first) as an FP16 pair, exactly, by
// the same trick in FP16: 0x6400 is 1024, whose last mantissa bits are free.
__device__ __forceinline__ __half2 NibblePairToHalf2(unsigned byte) {
    const unsigned bits = (byte & 0x0Fu) | ((byte & 0xF0u) << 12) | 0x64006400u;
    return __hsub2(*reinterpret_cast<const __half2*>(&bits), __float2half2_rn(1024.0f));
}

// An INT4 group's packed {scale, min} (the scale in the low half).
__device__ __forceinline__ float2 GroupRange(std::uint32_t packed) {
    return __half22float2(*reinterpret_cast<const __half2*>(&packed));
}

// One weight, decoded to FP32. For the paths that touch few weights (an
// embedding row) or don't need speed; the matrix kernels unpack in bulk.
__device__ __forceinline__ float DecodeWeight(const WeightView& w, int row, int col) {
    const std::size_t index = static_cast<std::size_t>(row) * w.in_features + col;
    switch (w.format) {
    case WeightFormat::kBf16:
        return Bf16ToFloat(reinterpret_cast<const Bf16Bits*>(w.data)[index]);
    case WeightFormat::kFp8:
        return __half2float(__half(__nv_cvt_fp8_to_halfraw(w.data[index], __NV_E4M3))) * w.row_scales[row];
    case WeightFormat::kInt4: {
        const unsigned byte = w.data[index / 2];
        const float2 range = GroupRange(w.group_ranges[index / kInt4GroupSize]);
        return static_cast<float>((col & 1) != 0 ? byte >> 4 : byte & 0xFu) * range.x + range.y;
    }
    }
    return 0.0f;
}

}  // namespace murmur
