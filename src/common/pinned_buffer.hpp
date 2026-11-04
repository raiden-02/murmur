#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "common/cuda_check.hpp"

namespace murmur {

// Page-locked ("pinned") CPU memory: the GPU can copy to and from it directly
// and asynchronously, and a copy from it can be recorded in a CUDA graph and
// replayed later with whatever the memory holds then. Move-only.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t count);
    ~PinnedBuffer();

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    T* Data() { return data_; }
    const T* Data() const { return data_; }
    std::span<T> Span() { return {data_, count_}; }
    std::size_t Size() const { return count_; }
    std::size_t SizeBytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// Allocates `count` elements of pinned memory, value-initialized. Throws
// std::length_error on size overflow and std::runtime_error on failure.
template <typename T>
PinnedBuffer<T>::PinnedBuffer(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::length_error("PinnedBuffer: count * sizeof(T) overflows");
    }
    if (count > 0) {
        void* memory = nullptr;
        CudaCheck(cudaMallocHost(&memory, count * sizeof(T)));
        data_ = static_cast<T*>(memory);
        count_ = count;
        for (std::size_t i = 0; i < count_; ++i) {
            data_[i] = T{};
        }
    }
}

// Never throws (see DeviceBuffer's destructor).
template <typename T>
PinnedBuffer<T>::~PinnedBuffer() {
    static_cast<void>(cudaFreeHost(data_));
}

template <typename T>
PinnedBuffer<T>::PinnedBuffer(PinnedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

template <typename T>
PinnedBuffer<T>& PinnedBuffer<T>::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        static_cast<void>(cudaFreeHost(data_));
        data_ = std::exchange(other.data_, nullptr);
        count_ = std::exchange(other.count_, 0);
    }
    return *this;
}

}  // namespace murmur
