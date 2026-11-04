#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace murmur {

void Argmax(const float* values, int count, std::int32_t* out_index, cudaStream_t stream);

}  // namespace murmur
