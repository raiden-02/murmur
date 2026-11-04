#pragma once

#include "engine/kernels/weight_view.cuh"

#include <cuda_runtime.h>

namespace murmur {

// Work fused into a matrix-vector product, so it costs no extra kernel launch
// or pass over memory.
struct MatVecOptions {
    const float* norm_weight = nullptr;   // RMSNorm x first: x_i <- norm_weight_i * x_i / rms(x)
    float norm_epsilon = 0.0f;
    const float* bias = nullptr;          // add bias (not with swiglu)
    bool accumulate = false;              // y += result: a fused residual add (not with swiglu)
    bool swiglu = false;                  // W interleaves gate and up rows: y_o = SiLU(gate_o . x) * (up_o . x)
};

void MatVec(const WeightView& weight, const float* x, float* y, const MatVecOptions& options, cudaStream_t stream);

}  // namespace murmur
