#pragma once

#include "common/device_buffer.hpp"
#include "engine/blas.hpp"
#include "engine/channel_importance.hpp"
#include "engine/kernels/gemm.cuh"
#include "engine/kernels/matvec.cuh"
#include "engine/weight_matrix.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace murmur {

// Scratch sizes for the multi-row path.
struct LinearLimits {
    int max_rows = 0;                      // rows per call
    int max_in_features = 0;               // widest x
    int max_norm_width = 0;                // widest x that is RMS-normalized first
    int max_swiglu_rows = 0;               // tallest gate/up matrix
    std::size_t max_matrix_elements = 0;   // largest matrix multiplied with several rows (exact path)
};

// How several rows at once are multiplied by a weight matrix.
enum class MultiRowMath {
    kExactFp32,     // weights decoded to FP32, cuBLAS SGEMM: the same arithmetic as one row
    kTensorCores,   // x rounded to FP16, GemmF16 unpacking the weights on tensor cores: any format
    kCublasBf16,    // x rounded to BF16, cuBLAS's BF16 GEMM: BF16 weights (others use kTensorCores)
    kFp8TensorCores,   // x rounded to FP8 too, cuBLASLt's FP8 GEMM: FP8 weights (others use kTensorCores)
    kFastest,       // the fastest of the above on tensor cores (measured with `murmur-lab batch`): GemmF16 for
                    // decode batches (up to kGemmF16Rows rows, bound by reading the weights); for prompt chunks
                    // (bound by the math) cuBLAS for BF16, FP8 tensor cores for FP8, GemmF16 for INT4
};

inline constexpr int kGemmF16Rows = 48;

std::string_view MultiRowMathName(MultiRowMath math);
std::optional<MultiRowMath> ParseMultiRowMath(std::string_view name);

// Linear layers with weights in any WeightFormat and FP32 activations, with
// the fusions of MatVecOptions (RMSNorm before, bias / residual add / SwiGLU
// after). One row runs the fused matrix-vector kernel. Several rows run the
// same steps around a matrix product: exact (RMSNorm into scratch, weights
// decoded to FP32, SGEMM) or on tensor cores (RMSNorm or rounding into
// 16-bit scratch, then GemmF16 with the epilogue fused, or cuBLAS followed
// by the bias or SwiGLU).
class Linear {
public:
    Linear(cudaStream_t stream, const LinearLimits& limits, MultiRowMath math);

    void Apply(const float* x, const WeightMatrix& weight, float* y, int rows, const MatVecOptions& options = {});
    void FeedForward(float* x, const WeightMatrix& gate_up, const WeightMatrix& down, float* activation, int rows,
                     const float* norm_weight, float norm_epsilon);
    void SetMultiRowMath(MultiRowMath math);
    void SetInputStatistics(InputStatistics* statistics) { statistics_ = statistics; }
    MultiRowMath GetMultiRowMath() const { return math_; }

private:
    bool UsesCublasBf16(const WeightMatrix& weight, int rows) const;
    void ApplyExact(const float* x, const WeightMatrix& weight, float* y, int rows, const MatVecOptions& options);
    void ApplyTensorCores(const float* x, const WeightMatrix& weight, float* y, int rows,
                          const MatVecOptions& options);
    void ApplyCublasBf16(const float* x, const WeightMatrix& weight, float* y, int rows,
                         const MatVecOptions& options);
    void ApplyFp8TensorCores(const float* x, const WeightMatrix& weight, float* y, int rows,
                             const MatVecOptions& options);
    void Finish(float* y, int rows, int out, const MatVecOptions& options);

    cudaStream_t stream_;
    LinearLimits limits_;
    MultiRowMath math_;
    InputStatistics* statistics_ = nullptr;   // calibration only: records every input
    Blas blas_;
    GemmWorkspace gemm_workspace_;
    DeviceBuffer<float> widened_;           // one weight matrix as FP32, allocated for exact math only
    DeviceBuffer<float> normed_;            // max_rows x max_norm_width
    DeviceBuffer<std::uint16_t> input16_;   // max_rows x max_in_features, FP16 or BF16
    DeviceBuffer<float> gate_up_;           // max_rows x max_swiglu_rows
    DeviceBuffer<std::uint8_t> input8_;     // max_rows x max_in_features, E4M3
    DeviceBuffer<float> input8_scales_;     // max_rows
};

}  // namespace murmur
