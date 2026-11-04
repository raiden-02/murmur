#pragma once

#include <cuda_runtime.h>

namespace murmur {

class GpuTimer {
public:
    GpuTimer();
    ~GpuTimer();

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void Start();
    float StopMilliseconds();

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

}  // namespace murmur
