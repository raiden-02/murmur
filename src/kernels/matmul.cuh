#pragma once

#include "common/device_buffer.hpp"

#include <cublas_v2.h>

namespace murmur {

enum class MatmulVersion {
    kNaive,
    kTiled,
};

const char* MatmulVersionName(MatmulVersion version);

void Matmul(const DeviceBuffer<float>& a, const DeviceBuffer<float>& b, DeviceBuffer<float>& c,
            int m, int n, int k, MatmulVersion version);

class CublasMatmul {
public:
    CublasMatmul();
    ~CublasMatmul();

    CublasMatmul(const CublasMatmul&) = delete;
    CublasMatmul& operator=(const CublasMatmul&) = delete;

    void Run(const DeviceBuffer<float>& a, const DeviceBuffer<float>& b, DeviceBuffer<float>& c,
             int m, int n, int k);

private:
    cublasHandle_t handle_ = nullptr;
};

}  // namespace murmur
