#pragma once

#include "common/device_buffer.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>

namespace murmur {

// Owns cuBLAS and cuBLASLt handles bound to one stream and runs the model's linear layers.
class Blas {
public:
    explicit Blas(cudaStream_t stream);
    ~Blas();

    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;

    void Linear(const float* x, const float* weight, float* y, int rows, int in_features, int out_features,
                bool accumulate = false);
    void LinearBf16(const std::uint16_t* x, const std::uint16_t* weight, float* y, int rows, int in_features,
                    int out_features, bool accumulate = false);
    void LinearFp8(const std::uint8_t* x, const std::uint8_t* weight, float* y, int rows, int in_features,
                   int out_features);

private:
    struct DescDestroy {
        void operator()(cublasLtMatmulDesc_t desc) const { static_cast<void>(cublasLtMatmulDescDestroy(desc)); }
    };
    struct LayoutDestroy {
        void operator()(cublasLtMatrixLayout_t layout) const {
            static_cast<void>(cublasLtMatrixLayoutDestroy(layout));
        }
    };
    using Desc = std::unique_ptr<std::remove_pointer_t<cublasLtMatmulDesc_t>, DescDestroy>;
    using Layout = std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>, LayoutDestroy>;

    // A cuBLASLt product's descriptors and chosen algorithm, made once per shape.
    struct Fp8Plan {
        Desc operation;
        Layout weight;
        Layout input;
        Layout output;
        cublasLtMatmulAlgo_t algorithm{};
    };
    const Fp8Plan& PlanFp8(int rows, int in_features, int out_features);

    cudaStream_t stream_;
    cublasHandle_t handle_ = nullptr;
    cublasLtHandle_t lt_ = nullptr;
    DeviceBuffer<std::uint8_t> workspace_;
    std::map<std::tuple<int, int, int>, Fp8Plan> fp8_plans_;
};

}  // namespace murmur
