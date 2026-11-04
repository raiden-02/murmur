#include "engine/linear.hpp"

#include "engine/kernels/elementwise.cuh"
#include "engine/kernels/quantize.cuh"
#include "engine/kernels/rms_norm.cuh"

#include <stdexcept>

namespace murmur {

std::string_view MultiRowMathName(MultiRowMath math) {
    switch (math) {
    case MultiRowMath::kExactFp32:
        return "exact";
    case MultiRowMath::kTensorCores:
        return "gemm";
    case MultiRowMath::kCublasBf16:
        return "cublas";
    case MultiRowMath::kFp8TensorCores:
        return "fp8";
    case MultiRowMath::kFastest:
        return "fast";
    }
    return "?";
}

std::optional<MultiRowMath> ParseMultiRowMath(std::string_view name) {
    for (const MultiRowMath math : {MultiRowMath::kExactFp32, MultiRowMath::kTensorCores, MultiRowMath::kCublasBf16,
                                    MultiRowMath::kFp8TensorCores, MultiRowMath::kFastest}) {
        if (name == MultiRowMathName(math)) {
            return math;
        }
    }
    return std::nullopt;
}

Linear::Linear(cudaStream_t stream, const LinearLimits& limits, MultiRowMath math)
    : stream_(stream),
      limits_(limits),
      math_(math),
      blas_(stream),
      normed_(static_cast<std::size_t>(limits.max_rows) * limits.max_norm_width),
      input16_(static_cast<std::size_t>(limits.max_rows) * limits.max_in_features),
      gate_up_(static_cast<std::size_t>(limits.max_rows) * limits.max_swiglu_rows),
      input8_(static_cast<std::size_t>(limits.max_rows) * limits.max_in_features),
      input8_scales_(limits.max_rows) {
    SetMultiRowMath(math);
}

// Exact math needs the widening scratch: allocated here, not on first use,
// because that first use may be inside a CUDA graph being recorded, where
// allocating is not allowed.
void Linear::SetMultiRowMath(MultiRowMath math) {
    math_ = math;
    if (math == MultiRowMath::kExactFp32 && widened_.Size() == 0) {
        widened_ = DeviceBuffer<float>(limits_.max_matrix_elements);
    }
}

// y = (x . W^T) with the fused work in `options`, for `rows` rows of x
// (rows x in_features, FP32 row-major). y is rows x out_features, or rows x
// out_features / 2 with swiglu.
//
// On the exact path, a matrix bigger than the widening scratch (only the
// vocabulary projection, and only when every row's logits are requested)
// runs one fused matrix-vector product per row instead; so does a matrix
// the tensor-core kernel can't tile.
void Linear::Apply(const float* x, const WeightMatrix& weight, float* y, int rows, const MatVecOptions& options) {
    const int in = weight.InFeatures();
    const int out = options.swiglu ? weight.OutFeatures() / 2 : weight.OutFeatures();
    if (rows < 0 || rows > limits_.max_rows) {
        throw std::invalid_argument("Linear::Apply: row count out of range");
    }
    if (rows == 0) {
        return;
    }
    if (statistics_ != nullptr) {
        statistics_->Add(weight, x, rows, options, stream_);
    }
    const bool exact = math_ == MultiRowMath::kExactFp32;
    if (rows == 1 || (exact && weight.ElementCount() > limits_.max_matrix_elements) ||
        (!exact && !GemmF16Supports(weight.View()))) {
        for (int r = 0; r < rows; ++r) {
            MatVec(weight.View(), x + static_cast<std::size_t>(r) * in, y + static_cast<std::size_t>(r) * out,
                   options, stream_);
        }
        return;
    }
    if (options.norm_weight != nullptr && in > limits_.max_norm_width) {
        throw std::invalid_argument("Linear::Apply: x too wide for the normalization scratch");
    }
    if (options.swiglu && weight.OutFeatures() > limits_.max_swiglu_rows) {
        throw std::invalid_argument("Linear::Apply: gate/up matrix too tall for the SwiGLU scratch");
    }
    if (!exact && in > limits_.max_in_features) {
        throw std::invalid_argument("Linear::Apply: x too wide for the 16-bit scratch");
    }
    const bool fastest = math_ == MultiRowMath::kFastest && rows > kGemmF16Rows;
    if (exact) {
        ApplyExact(x, weight, y, rows, options);
    } else if (UsesCublasBf16(weight, rows)) {
        ApplyCublasBf16(x, weight, y, rows, options);
    } else if ((math_ == MultiRowMath::kFp8TensorCores || fastest) && weight.Format() == WeightFormat::kFp8 &&
               weight.InFeatures() % 16 == 0 && weight.OutFeatures() % 16 == 0) {
        ApplyFp8TensorCores(x, weight, y, rows, options);
    } else {
        ApplyTensorCores(x, weight, y, rows, options);
    }
}

// A transformer's feed-forward block on the residual stream x:
// x += SwiGLU(RMSNorm(x) . Wgate_up^T) . Wdown^T, leaving the SwiGLU output
// in `activation` (rows x down's in_features). The same as Apply twice,
// except on the cuBLAS BF16 path, where the SwiGLU pass also writes its
// output rounded to BF16 for the down projection: one pass over the
// activation fewer.
void Linear::FeedForward(float* x, const WeightMatrix& gate_up, const WeightMatrix& down, float* activation, int rows,
                         const float* norm_weight, float norm_epsilon) {
    const MatVecOptions gate_up_options{.norm_weight = norm_weight, .norm_epsilon = norm_epsilon, .swiglu = true};
    if (statistics_ != nullptr || !UsesCublasBf16(gate_up, rows) || !UsesCublasBf16(down, rows) ||
        gate_up.InFeatures() > limits_.max_norm_width || gate_up.OutFeatures() > limits_.max_swiglu_rows ||
        down.InFeatures() != gate_up.OutFeatures() / 2 || down.InFeatures() > limits_.max_in_features) {
        Apply(x, gate_up, activation, rows, gate_up_options);
        Apply(activation, down, x, rows, {.accumulate = true});
        return;
    }
    const int hidden = gate_up.InFeatures();
    const int width = down.InFeatures();
    RmsNormToBf16(x, norm_weight, input16_.Data(), rows, hidden, norm_epsilon, stream_);
    blas_.LinearBf16(input16_.Data(), reinterpret_cast<const Bf16Bits*>(gate_up.View().data), gate_up_.Data(), rows,
                     hidden, gate_up.OutFeatures());
    SwiGluPairs(gate_up_.Data(), activation, rows, width, stream_, input16_.Data());
    blas_.LinearBf16(input16_.Data(), reinterpret_cast<const Bf16Bits*>(down.View().data), x, rows, width,
                     down.OutFeatures(), true);
}

// Whether Apply multiplies `rows` rows by `weight` with cuBLAS's BF16 GEMM.
bool Linear::UsesCublasBf16(const WeightMatrix& weight, int rows) const {
    const bool fastest = math_ == MultiRowMath::kFastest && rows > kGemmF16Rows;
    return rows > 1 && rows <= limits_.max_rows && weight.Format() == WeightFormat::kBf16 &&
           GemmF16Supports(weight.View()) && (math_ == MultiRowMath::kCublasBf16 || fastest);
}

void Linear::ApplyExact(const float* x, const WeightMatrix& weight, float* y, int rows, const MatVecOptions& options) {
    const int in = weight.InFeatures();
    const float* input = x;
    if (options.norm_weight != nullptr) {
        RmsNorm(x, options.norm_weight, normed_.Data(), rows, in, options.norm_epsilon, stream_);
        input = normed_.Data();
    }
    DequantizeRows(weight.View(), nullptr, weight.OutFeatures(), widened_.Data(), stream_);
    if (options.swiglu) {
        blas_.Linear(input, widened_.Data(), gate_up_.Data(), rows, in, weight.OutFeatures());
    } else {
        blas_.Linear(input, widened_.Data(), y, rows, in, weight.OutFeatures(), options.accumulate);
    }
    Finish(y, rows, options.swiglu ? weight.OutFeatures() / 2 : weight.OutFeatures(), options);
}

// RMSNorm or rounding into FP16 scratch, then one GemmF16 launch (two with
// split K) that also applies the bias, residual add or SwiGLU.
void Linear::ApplyTensorCores(const float* x, const WeightMatrix& weight, float* y, int rows,
                              const MatVecOptions& options) {
    const int in = weight.InFeatures();
    if (options.norm_weight != nullptr) {
        RmsNormToF16(x, options.norm_weight, input16_.Data(), rows, in, options.norm_epsilon, stream_);
    } else {
        ConvertF32ToF16(x, input16_.Data(), static_cast<std::size_t>(rows) * in, stream_);
    }
    MatVecOptions epilogue = options;
    epilogue.norm_weight = nullptr;
    GemmF16(input16_.Data(), rows, weight.View(), y, epilogue, gemm_workspace_, stream_);
}

void Linear::ApplyCublasBf16(const float* x, const WeightMatrix& weight, float* y, int rows,
                             const MatVecOptions& options) {
    const int in = weight.InFeatures();
    if (options.norm_weight != nullptr) {
        RmsNormToBf16(x, options.norm_weight, input16_.Data(), rows, in, options.norm_epsilon, stream_);
    } else {
        ConvertF32ToBf16(x, input16_.Data(), static_cast<std::size_t>(rows) * in, stream_);
    }
    const auto* w = reinterpret_cast<const Bf16Bits*>(weight.View().data);
    if (options.swiglu) {
        blas_.LinearBf16(input16_.Data(), w, gate_up_.Data(), rows, in, weight.OutFeatures());
    } else {
        blas_.LinearBf16(input16_.Data(), w, y, rows, in, weight.OutFeatures(), options.accumulate);
    }
    Finish(y, rows, options.swiglu ? weight.OutFeatures() / 2 : weight.OutFeatures(), options);
}

// RMSNorm and rounding into FP8 scratch with a scale per row, the product of
// FP8 codes on cuBLASLt, then one pass that applies both scales and the
// epilogue. Accumulating and SwiGLU products go through the gate/up scratch.
void Linear::ApplyFp8TensorCores(const float* x, const WeightMatrix& weight, float* y, int rows,
                                 const MatVecOptions& options) {
    const int in = weight.InFeatures();
    QuantizeRowsToFp8(x, rows, in, options.norm_weight, options.norm_epsilon, input8_.Data(), input8_scales_.Data(),
                      stream_);
    float* products = options.accumulate || options.swiglu ? gate_up_.Data() : y;
    if (products == gate_up_.Data() && weight.OutFeatures() > limits_.max_swiglu_rows) {
        throw std::invalid_argument("Linear::Apply: products too wide for the scratch");
    }
    blas_.LinearFp8(input8_.Data(), weight.View().data, products, rows, in, weight.OutFeatures());
    MatVecOptions epilogue = options;
    epilogue.norm_weight = nullptr;
    FinishFp8Products(products, rows, input8_scales_.Data(), weight.View(), y, epilogue, stream_);
}

// The epilogues of the cuBLAS paths: SwiGLU of the interleaved gate and up
// products, or the bias.
void Linear::Finish(float* y, int rows, int out, const MatVecOptions& options) {
    if (options.swiglu) {
        SwiGluPairs(gate_up_.Data(), y, rows, out, stream_);
    } else if (options.bias != nullptr) {
        AddBias(y, options.bias, rows, out, stream_);
    }
}

}  // namespace murmur
