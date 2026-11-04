#pragma once

#include "common/device_buffer.hpp"
#include "engine/kernels/matvec.cuh"
#include "engine/kernels/weight_view.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace murmur {

// Scratch for the split-K partial sums of GemmF16, allocated once: launches
// never allocate, so they can be recorded into CUDA graphs.
class GemmWorkspace {
public:
    GemmWorkspace();

    float* Partials() { return partials_.Data(); }
    std::size_t Capacity() const { return partials_.Size(); }
    int SmCount() const { return sm_count_; }

private:
    int sm_count_;
    DeviceBuffer<float> partials_;
};

void GemmF16(const std::uint16_t* x, int rows, const WeightView& weight, float* y, const MatVecOptions& options,
             GemmWorkspace& workspace, cudaStream_t stream);

bool GemmF16Supports(const WeightView& weight);

void FinishFp8Products(const float* products, int rows, const float* x_scales, const WeightView& weight, float* y,
                       const MatVecOptions& options, cudaStream_t stream);

}  // namespace murmur
