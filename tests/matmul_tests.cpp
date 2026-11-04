#include "kernels/matmul.cuh"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using murmur::MatmulVersion;

constexpr MatmulVersion kAllVersions[] = { MatmulVersion::kNaive, MatmulVersion::kTiled };

// Multiplies random m x k and k x n matrices on the GPU and compares every
// element of C with a double-precision CPU result. Float sums of k products
// round a little differently from the CPU's, so the tolerance scales with k
// and the size of the products.
void ExpectMatchesCpu(int m, int n, int k, MatmulVersion version) {
    std::mt19937 random(m * 131 + n * 17 + k);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> host_a(static_cast<std::size_t>(m) * k);
    std::vector<float> host_b(static_cast<std::size_t>(k) * n);
    for (float& v : host_a) v = distribution(random);
    for (float& v : host_b) v = distribution(random);

    murmur::DeviceBuffer<float> a(host_a.size());
    murmur::DeviceBuffer<float> b(host_b.size());
    murmur::DeviceBuffer<float> c(static_cast<std::size_t>(m) * n);
    a.CopyFromHost(host_a);
    b.CopyFromHost(host_b);
    murmur::Matmul(a, b, c, m, n, k, version);
    std::vector<float> host_c(c.Size());
    c.CopyToHost(host_c);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double expected = 0.0;
            double magnitude = 0.0;
            for (int i = 0; i < k; ++i) {
                const double product = static_cast<double>(host_a[static_cast<std::size_t>(row) * k + i]) *
                                       host_b[static_cast<std::size_t>(i) * n + col];
                expected += product;
                magnitude += std::abs(product);
            }
            ASSERT_NEAR(host_c[static_cast<std::size_t>(row) * n + col], expected, 1e-5 * k * (1.0 + magnitude / k) + 1e-6)
                << murmur::MatmulVersionName(version) << ", " << m << "x" << n << "x" << k
                << ", at C[" << row << "][" << col << "]";
        }
    }
}

}  // namespace

TEST(MatmulTest, OneByOne) {
    for (MatmulVersion version : kAllVersions) ExpectMatchesCpu(1, 1, 1, version);
}

// Exactly one 16 x 16 tile in every dimension.
TEST(MatmulTest, OneFullTile) {
    for (MatmulVersion version : kAllVersions) ExpectMatchesCpu(16, 16, 16, version);
}

// Several tiles along k: the tiled loop must add up every step.
TEST(MatmulTest, ManyTilesAlongK) {
    for (MatmulVersion version : kAllVersions) ExpectMatchesCpu(32, 48, 160, version);
}

// Sizes that aren't multiples of 16: edge tiles must load 0 outside the
// matrices and must not write outside C.
TEST(MatmulTest, PartialEdgeTiles) {
    for (MatmulVersion version : kAllVersions) {
        ExpectMatchesCpu(17, 33, 9, version);
        ExpectMatchesCpu(100, 70, 45, version);
    }
}

// A matrix-vector product (n = 1): the shape used when generating one token.
TEST(MatmulTest, MatrixTimesVector) {
    for (MatmulVersion version : kAllVersions) ExpectMatchesCpu(300, 1, 257, version);
}

TEST(MatmulTest, SizeMismatchThrows) {
    murmur::DeviceBuffer<float> a(6);
    murmur::DeviceBuffer<float> b(6);
    murmur::DeviceBuffer<float> c(4);
    EXPECT_THROW(murmur::Matmul(a, b, c, 2, 2, 4, MatmulVersion::kNaive), std::invalid_argument);
}

// The cuBLAS reference must agree with the naive kernel.
TEST(MatmulTest, CublasMatchesNaive) {
    constexpr int kM = 64, kN = 96, kK = 80;
    std::mt19937 random(3);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> host_a(kM * kK), host_b(kK * kN);
    for (float& v : host_a) v = distribution(random);
    for (float& v : host_b) v = distribution(random);
    murmur::DeviceBuffer<float> a(host_a.size()), b(host_b.size()), c1(kM * kN), c2(kM * kN);
    a.CopyFromHost(host_a);
    b.CopyFromHost(host_b);

    murmur::Matmul(a, b, c1, kM, kN, kK, MatmulVersion::kNaive);
    murmur::CublasMatmul cublas;
    cublas.Run(a, b, c2, kM, kN, kK);

    std::vector<float> r1(kM * kN), r2(kM * kN);
    c1.CopyToHost(r1);
    c2.CopyToHost(r2);
    for (std::size_t i = 0; i < r1.size(); ++i) {
        ASSERT_NEAR(r1[i], r2[i], 1e-3) << "at " << i;
    }
}
