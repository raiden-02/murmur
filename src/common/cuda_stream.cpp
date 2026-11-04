#include "common/cuda_stream.hpp"

#include "common/cuda_check.hpp"

#include <utility>

namespace murmur {

// Creates a blocking stream: it stays ordered with the legacy default
// stream, so plain synchronous copies such as DeviceBuffer::CopyFromHost
// can never race with kernels queued here.
CudaStream::CudaStream() {
    CudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamDefault));
}

// Never throws (see DeviceBuffer's destructor). Destroying a stream with work
// still queued is allowed: the work completes, then the stream is released.
CudaStream::~CudaStream() {
    if (stream_ != nullptr) {
        static_cast<void>(cudaStreamDestroy(stream_));
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
        }
        stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
}

// Blocks the calling CPU thread until everything queued so far has finished.
void CudaStream::Synchronize() const {
    CudaCheck(cudaStreamSynchronize(stream_));
}

// Hands everything queued so far to the GPU now, without waiting for it.
// On Windows the display driver (WDDM) holds queued work in a buffer until
// something forces it out, usually the next synchronize: work enqueued
// ahead of CPU-side bookkeeping would sit there while the GPU idles.
// Asking whether the stream is done is such a trigger.
void CudaStream::Flush() const {
    const cudaError_t status = cudaStreamQuery(stream_);
    if (status != cudaErrorNotReady) {
        CudaCheck(status);
    }
}

}  // namespace murmur
