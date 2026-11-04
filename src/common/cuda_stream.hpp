#pragma once

#include <cuda_runtime.h>

namespace murmur {

// Owns a CUDA stream: a queue of GPU work that runs in order. Move-only.
class CudaStream {
public:
    CudaStream();
    ~CudaStream();

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    cudaStream_t Get() const { return stream_; }
    void Synchronize() const;
    void Flush() const;

private:
    cudaStream_t stream_ = nullptr;
};

}  // namespace murmur
