#include "kernels/matmul.cuh"

#include "common/cuda_check.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace murmur {

namespace {

// Each block computes one kTile x kTile square of C with kTile x kTile threads,
// one thread per element of C.
constexpr int kTile = 16;

// Naive: each thread computes one element of C as the dot product of a row
// of A and a column of B, reading all 2 x k values straight from global
// memory. Neighbouring threads reread the same A row and B column values
// over and over, so most of that traffic is repeated.
__global__ void MatmulNaiveKernel(const float* a, const float* b, float* c, int m, int n, int k) {
    // kTile here is only the block size, 16: which cell of C this thread owns.
    // It is not a step along k. The loop below still walks all of k, one
    // element at a time, from global memory.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += a[static_cast<std::size_t>(row) * k + i] * b[static_cast<std::size_t>(i) * n + col];
    }
    c[static_cast<std::size_t>(row) * n + col] = sum;
}

// Tiled: the same result, but the block walks along k in steps of kTile.
// At each step it copies one kTile x kTile tile of A and one of B into shared
// memory, then every thread uses those 2 x kTile values from the fast table
// instead of global memory, so each loaded value is reused kTile times.
//
// Launched like the naive kernel: block (kTile, kTile), grid covering C.
// Thread (threadIdx.x, threadIdx.y) owns C[row][col] below.
__global__ void MatmulTiledKernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float sum = 0.0f;
    for (int tile_start = 0; tile_start < k; tile_start += kTile) {
        // Each thread copies one cell of each tile; cells outside A or B
        // become 0 so they add nothing to the dot product.
        const int a_col = tile_start + threadIdx.x;
        const int b_row = tile_start + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[static_cast<std::size_t>(row) * k + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[static_cast<std::size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();   // both tiles complete before anyone reads them
        for (int i = 0; i < kTile; ++i) {
            sum += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
        }
        __syncthreads();   // everyone done reading before the next load overwrites
    }
    if (row < m && col < n) {
        c[static_cast<std::size_t>(row) * n + col] = sum;
    }
}

}  // namespace

const char* MatmulVersionName(MatmulVersion version) {
    switch (version) {
    case MatmulVersion::kNaive: return "naive";
    case MatmulVersion::kTiled: return "tiled (shared memory)";
    }
    return "unknown";
}

// C = A x B for row-major matrices: A is m x k, B is k x n, C is m x n.
// C[row][col] = sum over i of A[row][i] * B[i][col].
void Matmul(const DeviceBuffer<float>& a, const DeviceBuffer<float>& b, DeviceBuffer<float>& c,
            int m, int n, int k, MatmulVersion version) {
    if (m < 0 || n < 0 || k < 0) {
        throw std::invalid_argument("Matmul: negative dimensions");
    }
    if (a.Size() != static_cast<std::size_t>(m) * k || b.Size() != static_cast<std::size_t>(k) * n ||
        c.Size() != static_cast<std::size_t>(m) * n) {
        throw std::invalid_argument("Matmul: buffer sizes don't match m, n, k");
    }
    if (m == 0 || n == 0) {
        return;
    }

    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    if (grid.y > 65535) {
        throw std::length_error("Matmul: too many rows for one launch");
    }
    switch (version) {
    case MatmulVersion::kNaive:
        MatmulNaiveKernel<<<grid, block>>>(a.Data(), b.Data(), c.Data(), m, n, k);
        break;
    case MatmulVersion::kTiled:
        MatmulTiledKernel<<<grid, block>>>(a.Data(), b.Data(), c.Data(), m, n, k);
        break;
    }
    CheckKernelLaunch();
}

namespace {

void CublasCheck(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed: cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

}  // namespace

// cuBLAS (NVIDIA's tuned linear-algebra library) needs a handle: its own
// per-thread state, created once and reused. This class owns it.
CublasMatmul::CublasMatmul() {
    CublasCheck(cublasCreate(&handle_), "cublasCreate");
}

CublasMatmul::~CublasMatmul() {
    static_cast<void>(cublasDestroy(handle_));
}

// Same C = A x B as Matmul, using cuBLAS as the speed reference.
// cuBLAS assumes column-major storage (column after column). A row-major
// matrix read as column-major is its transpose, so we ask cuBLAS for
// C^T = B^T x A^T, which leaves C in row-major order.
void CublasMatmul::Run(const DeviceBuffer<float>& a, const DeviceBuffer<float>& b,
                       DeviceBuffer<float>& c, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CublasCheck(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b.Data(), n,
                            a.Data(), k, &beta, c.Data(), n),
                "cublasSgemm");
}

}  // namespace murmur
