#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace murmur {

void RmsNorm(const float* in, const float* weight, float* out, int rows, int width, float epsilon,
             cudaStream_t stream);

void RmsNormToBf16(const float* in, const float* weight, std::uint16_t* out, int rows, int width, float epsilon,
                   cudaStream_t stream);

void RmsNormToF16(const float* in, const float* weight, std::uint16_t* out, int rows, int width, float epsilon,
                  cudaStream_t stream);

}  // namespace murmur
