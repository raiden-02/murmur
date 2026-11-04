#include "common/gpu_timer.hpp"

#include "common/cuda_check.hpp"

namespace murmur {

// Creates the two CUDA events (GPU-side timestamps) that bracket the timed
// work. If the second creation fails, the first event is destroyed before
// rethrowing: the destructor doesn't run for a half-built object.
GpuTimer::GpuTimer() {
    CudaCheck(cudaEventCreate(&start_));
    try {
        CudaCheck(cudaEventCreate(&stop_));
    } catch (...) {
        static_cast<void>(cudaEventDestroy(start_));
        throw;
    }
}

// Never throws (see DeviceBuffer's destructor for why).
GpuTimer::~GpuTimer() {
    static_cast<void>(cudaEventDestroy(stop_));
    static_cast<void>(cudaEventDestroy(start_));
}

// Queues a "start" timestamp on the GPU. It is recorded when the GPU reaches
// it, after all previously queued work, not when the CPU calls this.
void GpuTimer::Start() {
    CudaCheck(cudaEventRecord(start_));
}

// Queues a "stop" timestamp, waits for the GPU to reach it, and returns the
// GPU time between the two in milliseconds. CPU-side overhead between the
// calls is not included, only the GPU work queued in between.
float GpuTimer::StopMilliseconds() {
    CudaCheck(cudaEventRecord(stop_));
    CudaCheck(cudaEventSynchronize(stop_));
    float milliseconds = 0.0f;
    CudaCheck(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
}

}  // namespace murmur
