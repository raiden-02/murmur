#pragma once

#include "engine/kernels/weight_view.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace murmur {

void QuantizeToFp8(const Bf16Bits* in, int rows, int cols, std::uint8_t* out, float* row_scales, cudaStream_t stream);

void QuantizeToInt4(const Bf16Bits* in, int rows, int cols, const float* column_importance, std::uint8_t* out,
                    std::uint32_t* group_ranges, cudaStream_t stream);

void QuantizeRowsToFp8(const float* x, int rows, int width, const float* norm_weight, float epsilon, std::uint8_t* out,
                       float* row_scales, cudaStream_t stream);

void AccumulateInputSquares(const float* x, int rows, int width, const float* norm_weight, float epsilon, float* sums,
                            cudaStream_t stream);

void DequantizeRows(const WeightView& weight, const std::int32_t* row_ids, int count, float* out, cudaStream_t stream);

}  // namespace murmur
