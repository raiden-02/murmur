#pragma once

#include <cuda_runtime.h>
#include <source_location>

namespace murmur {

    void CudaCheck(cudaError_t error,
        std::source_location where = std::source_location::current());

    void CheckKernelLaunch(
        std::source_location where = std::source_location::current());
}
