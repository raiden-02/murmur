#include "common/device_info.hpp"

#include "common/cuda_check.hpp"

#include <cuda_runtime.h>

namespace murmur {

// The device's theoretical memory bandwidth in bytes per second: bus width
// times memory clock times 2 (GDDR moves data on both clock edges). The
// clock is reported in kHz.
double PeakMemoryBandwidth(int device) {
    int clock_khz = 0;
    int bus_width_bits = 0;
    CudaCheck(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device));
    CudaCheck(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device));
    return 2.0 * clock_khz * 1000.0 * (bus_width_bits / 8.0);
}

// The current device's SM count, read once: kernels size their grids by it
// on every launch, and the process never changes device.
int CurrentDeviceSmCount() {
    static const int count = [] {
        int device = 0;
        CudaCheck(cudaGetDevice(&device));
        int sms = 0;
        CudaCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        return sms;
    }();
    return count;
}

}  // namespace murmur
