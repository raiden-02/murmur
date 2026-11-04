#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace murmur {

void ConvertBf16ToF32(const std::uint16_t* in, float* out, std::size_t count, cudaStream_t stream);

void ConvertF32ToBf16(const float* in, std::uint16_t* out, std::size_t count, cudaStream_t stream);

void ConvertF32ToF16(const float* in, std::uint16_t* out, std::size_t count, cudaStream_t stream);

void AddBias(float* x, const float* bias, int rows, int width, cudaStream_t stream);

void SwiGluPairs(const float* gate_up, float* out, int rows, int width, cudaStream_t stream,
                 std::uint16_t* rounded = nullptr);

void GatherRows(const float* in, const int* row_indices, int count, int width, float* out, cudaStream_t stream);

}  // namespace murmur
