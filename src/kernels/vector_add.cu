#include "kernels/vector_add.cuh"

#include "common/cuda_check.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace murmur {

namespace {

constexpr unsigned int kThreadsPerBlock = 256;

// One thread per element. Each thread computes its global index from its
// block's position and its own position inside the block. Threads past the
// end (the last block is usually partly unused) do nothing.
__global__ void VectorAddKernel(const float* a, const float* b, float* c, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

}  // namespace

// c = a + b, element by element, on the GPU. All three buffers must have the
// same size. Returns as soon as the kernel is queued; the work finishes later
// (a copy back to the host waits for it).
void VectorAdd(const DeviceBuffer<float>& a, const DeviceBuffer<float>& b, DeviceBuffer<float>& c) {
    if (a.Size() != b.Size() || a.Size() != c.Size()) {
        throw std::invalid_argument("VectorAdd: all three vectors must have the same size");
    }
    const std::size_t n = a.Size();
    if (n == 0) {
        return;
    }

    // Round up so every element gets a thread.
    const std::size_t blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("VectorAdd: too many elements for one launch");
    }

    VectorAddKernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock>>>(
        a.Data(), b.Data(), c.Data(), n);
    CheckKernelLaunch();
}

}  // namespace murmur
