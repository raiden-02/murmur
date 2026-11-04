#include "engine/blas.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace murmur {

namespace {

constexpr std::size_t kWorkspaceBytes = 32 << 20;

void CublasCheck(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed: " + cublasGetStatusString(status));
    }
}

}  // namespace

// Creates the handle and binds it to `stream`, so every product queues
// behind the kernels already issued there. The default math mode keeps FP32
// products in full FP32 (no TF32 tensor-core shortcut), which the
// layer-by-layer comparison with the reference relies on. cuBLAS gets a
// fixed workspace of its own, so it never allocates while a CUDA graph is
// being recorded.
Blas::Blas(cudaStream_t stream) : stream_(stream), workspace_(kWorkspaceBytes) {
    CublasCheck(cublasCreate(&handle_), "cublasCreate");
    try {
        CublasCheck(cublasSetStream(handle_, stream), "cublasSetStream");
        CublasCheck(cublasSetWorkspace(handle_, workspace_.Data(), workspace_.SizeBytes()), "cublasSetWorkspace");
        CublasCheck(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
        CublasCheck(cublasLtCreate(&lt_), "cublasLtCreate");
    } catch (...) {
        static_cast<void>(cublasDestroy(handle_));
        throw;
    }
}

Blas::~Blas() {
    fp8_plans_.clear();   // before the handle they were made with
    static_cast<void>(cublasLtDestroy(lt_));
    static_cast<void>(cublasDestroy(handle_));
}

// A linear layer without bias: y = x . W^T (or y += x . W^T when
// `accumulate`), with
//   x: rows x in_features,  W: out_features x in_features (PyTorch's layout),
//   y: rows x out_features, all row-major.
//
// cuBLAS is column-major, and a row-major matrix read column-major is its
// transpose. So we ask for y^T = W . x^T: cuBLAS sees W as its transpose
// (in x out) and transposes it back with CUBLAS_OP_T; x^T and y^T need no
// work at all.
void Blas::Linear(const float* x, const float* weight, float* y, int rows, int in_features, int out_features,
                  bool accumulate) {
    if (rows < 0 || in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("Blas::Linear: bad shape");
    }
    if (rows == 0) {
        return;
    }
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    CublasCheck(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, out_features, rows, in_features, &alpha, weight,
                            in_features, x, in_features, &beta, y, out_features),
                "cublasSgemm");
}

// The same linear layer with BF16 x and W (raw 16-bit patterns) and an FP32
// result: cuBLAS runs it on tensor cores, multiplying BF16 pairs and adding
// the products in FP32. Reads the weights as they are stored, with no
// widened copy.
void Blas::LinearBf16(const std::uint16_t* x, const std::uint16_t* weight, float* y, int rows, int in_features,
                      int out_features, bool accumulate) {
    if (rows < 0 || in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("Blas::LinearBf16: bad shape");
    }
    if (rows == 0) {
        return;
    }
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    CublasCheck(cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, out_features, rows, in_features, &alpha, weight,
                             CUDA_R_16BF, in_features, x, CUDA_R_16BF, in_features, &beta, y, CUDA_R_32F,
                             out_features, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                "cublasGemmEx");
}

// The same linear layer on FP8 tensor cores: x and W both E4M3 (raw bytes),
// products summed in FP32, y FP32 (rows x out_features). Both inputs are
// unscaled codes, so y is off by x's row scales times W's row scales; the
// caller applies them (GemmF16's epilogue). FP8 runs at twice the FP16
// tensor-core rate, at the cost of rounding x to 4 significant bits.
void Blas::LinearFp8(const std::uint8_t* x, const std::uint8_t* weight, float* y, int rows, int in_features,
                     int out_features) {
    if (rows < 0 || in_features <= 0 || out_features <= 0 || in_features % 16 != 0 || out_features % 16 != 0) {
        throw std::invalid_argument("Blas::LinearFp8: FP8 products need positive multiples of 16");
    }
    if (rows == 0) {
        return;
    }
    const Fp8Plan& plan = PlanFp8(rows, in_features, out_features);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CublasCheck(cublasLtMatmul(lt_, plan.operation.get(), &alpha, weight, plan.weight.get(), x, plan.input.get(), &beta,
                               y, plan.output.get(), y, plan.output.get(), &plan.algorithm, workspace_.Data(),
                               workspace_.SizeBytes(), stream_),
                "cublasLtMatmul");
}

// Describes y^T = W . x^T for cuBLASLt (column-major, like Linear: W read
// transposed, the "TN" layout FP8 requires) and asks it for the fastest
// algorithm that fits the workspace. Made once per shape: the heuristic
// query costs far more than the product for a few rows.
const Blas::Fp8Plan& Blas::PlanFp8(int rows, int in_features, int out_features) {
    const auto key = std::make_tuple(rows, in_features, out_features);
    if (const auto it = fp8_plans_.find(key); it != fp8_plans_.end()) {
        return it->second;
    }
    Fp8Plan plan;
    cublasLtMatmulDesc_t operation = nullptr;
    CublasCheck(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
    plan.operation.reset(operation);
    const cublasOperation_t transpose = CUBLAS_OP_T;
    const cublasOperation_t keep = CUBLAS_OP_N;
    CublasCheck(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
                "cublasLtMatmulDescSetAttribute");
    CublasCheck(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &keep, sizeof(keep)),
                "cublasLtMatmulDescSetAttribute");
    const auto layout = [](cudaDataType type, int rows, int cols) {
        cublasLtMatrixLayout_t made = nullptr;
        CublasCheck(cublasLtMatrixLayoutCreate(&made, type, rows, cols, rows), "cublasLtMatrixLayoutCreate");
        return Layout(made);
    };
    plan.weight = layout(CUDA_R_8F_E4M3, in_features, out_features);
    plan.input = layout(CUDA_R_8F_E4M3, in_features, rows);
    plan.output = layout(CUDA_R_32F, out_features, rows);

    cublasLtMatmulPreference_t preference = nullptr;
    CublasCheck(cublasLtMatmulPreferenceCreate(&preference), "cublasLtMatmulPreferenceCreate");
    const std::unique_ptr<std::remove_pointer_t<cublasLtMatmulPreference_t>, void (*)(cublasLtMatmulPreference_t)>
        preference_owner(preference, [](cublasLtMatmulPreference_t p) {
            static_cast<void>(cublasLtMatmulPreferenceDestroy(p));
        });
    const std::size_t workspace_bytes = workspace_.SizeBytes();
    CublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_bytes, sizeof(workspace_bytes)),
                "cublasLtMatmulPreferenceSetAttribute");
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    const cublasStatus_t status =
        cublasLtMatmulAlgoGetHeuristic(lt_, operation, plan.weight.get(), plan.input.get(), plan.output.get(),
                                       plan.output.get(), preference, 1, &result, &found);
    if (status != CUBLAS_STATUS_SUCCESS || found == 0) {
        throw std::runtime_error("Blas::LinearFp8: cuBLASLt has no FP8 algorithm for this shape");
    }
    plan.algorithm = result.algo;
    return fp8_plans_.emplace(key, std::move(plan)).first->second;
}

}  // namespace murmur
