#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "common/cuda_check.hpp"

namespace murmur {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    T* Data();
    const T* Data() const;
    std::size_t Size() const;
    std::size_t SizeBytes() const;

    void CopyFromHost(std::span<const T> source);
    void CopyToHost(std::span<T> destination) const;

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// Allocates room for `count` elements of T in GPU global memory.
// Throws std::length_error if count * sizeof(T) would overflow, and
// std::runtime_error (via CudaCheck) if the GPU is out of memory.
// A count of 0 allocates nothing and leaves the buffer empty.
template <typename T>
DeviceBuffer<T>::DeviceBuffer(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::length_error("DeviceBuffer: count * sizeof(T) overflows");
    }
    if (count > 0) {
        CudaCheck(cudaMalloc(&data_, count * sizeof(T)));
        count_ = count;
    }
}

// Frees the GPU memory. Never throws: a destructor may run while another
// exception is already unwinding the stack, and a second one would
// terminate the program. cudaFree(nullptr) is a safe no-op, so empty and
// moved-from buffers need no special case.
template <typename T>
DeviceBuffer<T>::~DeviceBuffer() {
    static_cast<void>(cudaFree(data_));
}

// Takes over `other`'s allocation and leaves `other` empty, so exactly one
// object still owns the memory. No CUDA call: only a pointer moves.
template <typename T>
DeviceBuffer<T>::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      count_(std::exchange(other.count_, 0)) {}

// Frees this buffer's current memory, then takes over `other`'s and leaves
// `other` empty. Moving a buffer into itself does nothing.
template <typename T>
DeviceBuffer<T>& DeviceBuffer<T>::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        static_cast<void>(cudaFree(data_));
        data_ = std::exchange(other.data_, nullptr);
        count_ = std::exchange(other.count_, 0);
    }
    return *this;
}

// Device address of the first element, for passing to kernels. The CPU
// must not dereference it.
template <typename T>
T* DeviceBuffer<T>::Data() {
    return data_;
}

template <typename T>
const T* DeviceBuffer<T>::Data() const {
    return data_;
}

// Number of elements (not bytes).
template <typename T>
std::size_t DeviceBuffer<T>::Size() const {
    return count_;
}

template <typename T>
std::size_t DeviceBuffer<T>::SizeBytes() const {
    return count_ * sizeof(T);
}

// Copies `source` from CPU memory into this buffer. The sizes must match
// exactly, so a partial copy can't happen by accident. Blocks until the copy
// has finished.
template <typename T>
void DeviceBuffer<T>::CopyFromHost(std::span<const T> source) {
    if (source.size() != count_) {
        throw std::invalid_argument("DeviceBuffer::CopyFromHost: size mismatch");
    }
    if (count_ == 0) {
        return;
    }
    CudaCheck(cudaMemcpy(data_, source.data(), SizeBytes(), cudaMemcpyHostToDevice));
}

// Copies this buffer into `destination` in CPU memory. The sizes must match
// exactly. Blocks until the copy has finished, which also waits for any
// earlier kernel that writes this buffer.
template <typename T>
void DeviceBuffer<T>::CopyToHost(std::span<T> destination) const {
    if (destination.size() != count_) {
        throw std::invalid_argument("DeviceBuffer::CopyToHost: size mismatch");
    }
    if (count_ == 0) {
        return;
    }
    CudaCheck(cudaMemcpy(destination.data(), data_, SizeBytes(), cudaMemcpyDeviceToHost));
}

}  // namespace murmur
