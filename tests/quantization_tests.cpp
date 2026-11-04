#include "common/cuda_check.hpp"
#include "common/cuda_stream.hpp"
#include "common/device_buffer.hpp"
#include "engine/channel_importance.hpp"
#include "engine/kernels/gemm.cuh"
#include "engine/kernels/matvec.cuh"
#include "engine/kernels/quantize.cuh"
#include "engine/kv_cache.hpp"
#include "engine/weight_matrix.hpp"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace murmur {
namespace {

constexpr WeightFormat kFormats[] = {WeightFormat::kBf16, WeightFormat::kFp8, WeightFormat::kInt4};

float Bf16Value(Bf16Bits bits) {
    const std::uint32_t full = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &full, sizeof(value));
    return value;
}

// Random BF16 weights whose rows differ in magnitude by up to 10^4, as real
// rows do: per-row and per-group scales must cope with both.
std::vector<Bf16Bits> RandomWeights(int rows, int cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<Bf16Bits> bits(static_cast<std::size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        const float magnitude = std::pow(10.0f, static_cast<float>(r % 5) - 3.0f);
        for (int c = 0; c < cols; ++c) {
            const float value = dist(rng) * magnitude;
            std::uint32_t full = 0;
            std::memcpy(&full, &value, sizeof(full));
            bits[static_cast<std::size_t>(r) * cols + c] = static_cast<Bf16Bits>(full >> 16);
        }
    }
    return bits;
}

std::vector<float> RandomFloats(std::size_t count, unsigned seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> values(count);
    for (float& v : values) {
        v = dist(rng);
    }
    return values;
}

template <typename T>
DeviceBuffer<T> Upload(const std::vector<T>& host) {
    DeviceBuffer<T> device(host.size());
    device.CopyFromHost(host);
    return device;
}

template <typename T>
std::vector<T> Download(const DeviceBuffer<T>& device, const CudaStream& stream) {
    stream.Synchronize();
    std::vector<T> host(device.Size());
    device.CopyToHost(host);
    return host;
}

WeightMatrix MatrixFrom(const std::vector<Bf16Bits>& bits, int rows, int cols, WeightFormat format) {
    DeviceBuffer<std::uint8_t> bytes(bits.size() * sizeof(Bf16Bits));
    CudaCheck(cudaMemcpy(bytes.Data(), bits.data(), bytes.Size(), cudaMemcpyHostToDevice));
    return WeightMatrix::FromBf16(std::move(bytes), rows, cols, format);
}

// The weights a matrix actually holds, decoded to FP32 by the GPU.
std::vector<float> Decoded(const WeightMatrix& m, const CudaStream& stream) {
    DeviceBuffer<float> out(m.ElementCount());
    DequantizeRows(m.View(), nullptr, m.OutFeatures(), out.Data(), stream.Get());
    return Download(out, stream);
}

// Every element within `tolerance` relative to the largest reference magnitude.
void ExpectClose(const std::vector<float>& actual, const std::vector<double>& expected, double tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    double scale = 1e-30;
    for (const double e : expected) {
        scale = std::max(scale, std::abs(e));
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_NEAR(actual[i] / scale, expected[i] / scale, tolerance) << "at index " << i;
    }
}

class QuantizationTest : public ::testing::Test {
protected:
    CudaStream stream_;
};

TEST_F(QuantizationTest, FormatsTakeTheExpectedBytes) {
    constexpr int kRows = 64;
    constexpr int kCols = 1536;
    const std::vector<Bf16Bits> bits = RandomWeights(kRows, kCols, 1);
    const std::size_t n = static_cast<std::size_t>(kRows) * kCols;
    EXPECT_EQ(MatrixFrom(bits, kRows, kCols, WeightFormat::kBf16).SizeBytes(), 2 * n);
    EXPECT_EQ(MatrixFrom(bits, kRows, kCols, WeightFormat::kFp8).SizeBytes(), n + 4 * kRows);   // + a scale per row
    EXPECT_EQ(MatrixFrom(bits, kRows, kCols, WeightFormat::kInt4).SizeBytes(), n / 2 + 4 * n / kInt4GroupSize);
    EXPECT_THROW(MatrixFrom(RandomWeights(2, 100, 1), 2, 100, WeightFormat::kInt4), std::invalid_argument);
}

TEST_F(QuantizationTest, Fp8KeepsEveryWeightWithinHalfAStep) {
    // E4M3 has 3 mantissa bits: a step is 1/8 of the value's power of two, so
    // rounding is off by at most 1/16 relative. Below 2^-6 (scaled) values
    // are subnormal and the step stays at 2^-9: an absolute error of 2^-10.
    // The row's largest weight maps to 448 exactly.
    constexpr int kRows = 20;
    constexpr int kCols = 1536;
    const std::vector<Bf16Bits> bits = RandomWeights(kRows, kCols, 2);
    const WeightMatrix m = MatrixFrom(bits, kRows, kCols, WeightFormat::kFp8);
    const std::vector<float> decoded = Decoded(m, stream_);
    for (int r = 0; r < kRows; ++r) {
        float largest = 0.0f;
        for (int c = 0; c < kCols; ++c) {
            largest = std::max(largest, std::abs(Bf16Value(bits[r * kCols + c])));
        }
        const float scale = largest / 448.0f;
        for (int c = 0; c < kCols; ++c) {
            const float w = Bf16Value(bits[r * kCols + c]);
            const float bound = std::max(std::abs(w) / 16.0f, scale / 1024.0f) * 1.0001f;
            ASSERT_LE(std::abs(decoded[r * kCols + c] - w), bound) << "row " << r << " col " << c;
            if (std::abs(w) == largest) {
                EXPECT_NEAR(decoded[r * kCols + c], w, largest * 1e-6f);
            }
        }
    }
}

// Squared rounding error of one INT4 group on the grid spanning [low, high],
// computed the way the format stores it (step and min in FP16), each weight
// weighted by its column's importance.
double Int4GroupError(const float* w, const float* importance, float low, float high) {
    const float step = __half2float(__float2half_rn((high - low) / 15.0f));
    const float base = __half2float(__float2half_rn(low));
    double error = 0.0;
    for (int i = 0; i < kInt4GroupSize; ++i) {
        const float q = step > 0.0f ? std::clamp(std::rint((w[i] - base) / step), 0.0f, 15.0f) : 0.0f;
        const double d = q * step + base - w[i];
        error += importance[i] * d * d;
    }
    return error;
}

double DecodedError(const std::vector<float>& decoded, const std::vector<float>& w, const float* importance,
                    std::size_t group) {
    double error = 0.0;
    for (int i = 0; i < kInt4GroupSize; ++i) {
        const std::size_t index = group * kInt4GroupSize + i;
        const double d = decoded[index] - w[index];
        error += importance[i] * d * d;
    }
    return error;
}

std::vector<float> Values(const std::vector<Bf16Bits>& bits) {
    std::vector<float> values;
    for (const Bf16Bits b : bits) {
        values.push_back(Bf16Value(b));
    }
    return values;
}

TEST_F(QuantizationTest, Int4NeverRoundsWorseThanTheFullRange) {
    // Each group of 128 is cut into 15 equal steps. Spanning [min, max] keeps
    // every weight within half a step; clipping the range can do better
    // overall (outliers saturate, the rest get finer steps), and the
    // quantizer keeps whichever range has the smaller squared error, so it is
    // never worse than the full range.
    constexpr int kRows = 10;
    constexpr int kCols = 1024;
    const std::vector<float> w = Values(RandomWeights(kRows, kCols, 3));
    const std::vector<float> ones(kInt4GroupSize, 1.0f);
    const WeightMatrix m = MatrixFrom(RandomWeights(kRows, kCols, 3), kRows, kCols, WeightFormat::kInt4);
    const std::vector<float> decoded = Decoded(m, stream_);
    int clipped = 0;
    for (std::size_t g = 0; g < w.size() / kInt4GroupSize; ++g) {
        const float* group = w.data() + g * kInt4GroupSize;
        const auto [low, high] = std::minmax_element(group, group + kInt4GroupSize);
        const double full_range = Int4GroupError(group, ones.data(), *low, *high);
        const double chosen = DecodedError(decoded, w, ones.data(), g);
        ASSERT_LE(chosen, full_range * (1.0 + 1e-4)) << "group " << g;
        clipped += chosen < full_range * 0.999 ? 1 : 0;
    }
    EXPECT_GT(clipped, 0) << "random groups have outliers worth clipping";
}

TEST_F(QuantizationTest, Int4ImportanceProtectsTheColumnsThatMatter) {
    // With a few columns marked 1000x more important, the quantizer picks
    // ranges that round those columns better: their error drops, and the
    // importance-weighted error is never worse than without importance.
    constexpr int kRows = 16;
    constexpr int kCols = 512;
    const std::vector<Bf16Bits> bits = RandomWeights(kRows, kCols, 13);
    const std::vector<float> w = Values(bits);
    std::vector<float> importance(kCols, 1.0f);
    for (int c = 0; c < kCols; c += 16) {
        importance[c] = 1000.0f;
    }
    const std::vector<float> plain = Decoded(MatrixFrom(bits, kRows, kCols, WeightFormat::kInt4), stream_);
    DeviceBuffer<std::uint8_t> bytes(bits.size() * sizeof(Bf16Bits));
    CudaCheck(cudaMemcpy(bytes.Data(), bits.data(), bytes.Size(), cudaMemcpyHostToDevice));
    const std::vector<float> guided =
        Decoded(WeightMatrix::FromBf16(std::move(bytes), kRows, kCols, WeightFormat::kInt4, importance), stream_);
    double plain_important = 0.0;
    double guided_important = 0.0;
    for (std::size_t g = 0; g < w.size() / kInt4GroupSize; ++g) {
        const float* group_importance = importance.data() + (g % (kCols / kInt4GroupSize)) * kInt4GroupSize;
        ASSERT_LE(DecodedError(guided, w, group_importance, g), DecodedError(plain, w, group_importance, g) * (1.0 + 1e-4))
            << "group " << g;
    }
    for (std::size_t i = 0; i < w.size(); i += 16) {
        plain_important += std::pow(plain[i] - w[i], 2);
        guided_important += std::pow(guided[i] - w[i], 2);
    }
    EXPECT_LT(guided_important, 0.8 * plain_important);
}

TEST_F(QuantizationTest, InputStatisticsMeasureMeanSquaresAfterTheNorm) {
    constexpr int kRows = 7;
    constexpr int kWidth = 256;
    constexpr float kEpsilon = 1e-6f;
    const WeightMatrix m = MatrixFrom(RandomWeights(4, kWidth, 14), 4, kWidth, WeightFormat::kBf16);
    const auto x = RandomFloats(kRows * kWidth, 15, 3.0f);
    const auto norm = RandomFloats(kWidth, 16);
    const auto x_d = Upload(x);
    const auto norm_d = Upload(norm);
    InputStatistics statistics;
    statistics.Add(m, x_d.Data(), 3, {.norm_weight = norm_d.Data(), .norm_epsilon = kEpsilon}, stream_.Get());
    statistics.Add(m, x_d.Data() + 3 * kWidth, kRows - 3, {.norm_weight = norm_d.Data(), .norm_epsilon = kEpsilon},
                   stream_.Get());
    const std::vector<float> measured = statistics.MeanSquares(m);
    std::vector<double> expected(kWidth, 0.0);
    for (int r = 0; r < kRows; ++r) {
        double sum = 0.0;
        for (int c = 0; c < kWidth; ++c) {
            sum += static_cast<double>(x[r * kWidth + c]) * x[r * kWidth + c];
        }
        const double inverse_rms = 1.0 / std::sqrt(sum / kWidth + kEpsilon);
        for (int c = 0; c < kWidth; ++c) {
            expected[c] += std::pow(norm[c] * x[r * kWidth + c] * inverse_rms, 2) / kRows;
        }
    }
    ExpectClose(measured, expected, 1e-5);
    EXPECT_TRUE(statistics.MeanSquares(MatrixFrom(RandomWeights(4, kWidth, 17), 4, kWidth, WeightFormat::kBf16)).empty());
}

TEST_F(QuantizationTest, ChannelImportanceSurvivesSaveAndLoad) {
    ChannelImportance importance;
    importance.Set("layers.0.qkv", {1.0f, 2.5f, 1e-9f});
    importance.Set("lm_head", std::vector<float>(1536, 0.25f));
    const auto path = std::filesystem::temp_directory_path() / "murmur_importance_test.bin";
    importance.Save(path);
    const ChannelImportance loaded = ChannelImportance::Load(path);
    std::filesystem::remove(path);
    ASSERT_NE(loaded.Find("layers.0.qkv"), nullptr);
    EXPECT_EQ(*loaded.Find("layers.0.qkv"), (std::vector<float>{1.0f, 2.5f, 1e-9f}));
    EXPECT_EQ(*loaded.Find("lm_head"), std::vector<float>(1536, 0.25f));
    EXPECT_EQ(loaded.Find("layers.1.qkv"), nullptr);
    EXPECT_THROW(ChannelImportance::Load(std::filesystem::temp_directory_path() / "no_such_file.bin"), std::runtime_error);
}

TEST_F(QuantizationTest, KvCacheFormatsSizeEveryLayer) {
    // Qwen2.5-1.5B's shape: 28 layers, 2 KV heads of 128. A token takes 57 KB
    // in FP32 and half that in BF16; FP8 stores one byte per value plus a
    // scale per head, except in the first two layers, which keep BF16.
    ModelConfig config;
    config.num_layers = 28;
    config.num_kv_heads = 2;
    config.head_dim = 128;
    const std::size_t fp32 = PagedKvCache::BytesPerPage(config, 1, KvFormat::kFp32);
    const std::size_t bf16 = PagedKvCache::BytesPerPage(config, 1, KvFormat::kBf16);
    const std::size_t fp8 = PagedKvCache::BytesPerPage(config, 1, KvFormat::kFp8);
    EXPECT_EQ(fp32, 2u * 28 * 256 * 4);
    EXPECT_EQ(bf16, fp32 / 2);
    EXPECT_EQ(fp8, 2u * 2 * 256 * 2 + 2u * 26 * (256 + 2 * 4));
    EXPECT_EQ(PagedKvCache::BytesPerPage(config, 16, KvFormat::kFp8), 16 * fp8);

    PagedKvCache cache(config, 3, 16, KvFormat::kFp8);
    EXPECT_EQ(cache.SizeBytes(), 3 * 16 * fp8);
    EXPECT_EQ(cache.Layer(0).format, KvFormat::kBf16);
    EXPECT_EQ(cache.Layer(1).key_scales, nullptr);
    const KvLayer layer2 = cache.Layer(2);
    const KvLayer layer3 = cache.Layer(3);
    EXPECT_EQ(layer2.format, KvFormat::kFp8);
    ASSERT_NE(layer2.key_scales, nullptr);
    // Each layer's keys, values and scales sit back to back without overlap.
    EXPECT_EQ(static_cast<std::uint8_t*>(layer2.values) - static_cast<std::uint8_t*>(layer2.keys), 3 * 16 * 256);
    EXPECT_EQ(static_cast<std::uint8_t*>(layer3.keys) - static_cast<std::uint8_t*>(layer2.values), 3 * 16 * 256);
    EXPECT_EQ(layer2.value_scales - layer2.key_scales, 3 * 16 * 2);
    EXPECT_EQ(layer3.key_scales - layer2.value_scales, 3 * 16 * 2);
}

// y = W x (+ bias, + start) in double, with W given as decoded FP32 values.
std::vector<double> Reference(const std::vector<float>& x, const std::vector<float>& w, int rows, int in, int out,
                              const std::vector<float>& bias, const std::vector<float>& start) {
    std::vector<double> y(static_cast<std::size_t>(rows) * out);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < out; ++o) {
            double acc = start.empty() ? 0.0 : start[static_cast<std::size_t>(r) * out + o];
            for (int i = 0; i < in; ++i) {
                acc += static_cast<double>(x[static_cast<std::size_t>(r) * in + i]) * w[static_cast<std::size_t>(o) * in + i];
            }
            y[static_cast<std::size_t>(r) * out + o] = acc + (bias.empty() ? 0.0 : bias[o]);
        }
    }
    return y;
}

// SiLU(gate) * up over interleaved (gate, up) output pairs.
std::vector<double> SwiGlu(const std::vector<double>& gate_up) {
    std::vector<double> out(gate_up.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double g = gate_up[2 * i];
        out[i] = g / (1.0 + std::exp(-g)) * gate_up[2 * i + 1];
    }
    return out;
}

TEST_F(QuantizationTest, MatVecReadsEveryFormat) {
    // The decode kernel must compute with exactly the weights the format
    // holds (as DequantizeRows decodes them), on the model's widths.
    struct Shape {
        int out;
        int in;
    };
    for (const WeightFormat format : kFormats) {
        for (const Shape shape : {Shape{2048, 1536}, Shape{1536, 8960}, Shape{20000, 1536}}) {
            SCOPED_TRACE(std::string(WeightFormatName(format)) + " " + std::to_string(shape.out) + "x" +
                         std::to_string(shape.in));
            const WeightMatrix m = MatrixFrom(RandomWeights(shape.out, shape.in, 4), shape.out, shape.in, format);
            const std::vector<float> w = Decoded(m, stream_);
            const auto x = RandomFloats(shape.in, 5);
            const auto bias = RandomFloats(shape.out, 6);
            const auto x_d = Upload(x);
            const auto bias_d = Upload(bias);
            DeviceBuffer<float> y(shape.out);
            MatVec(m.View(), x_d.Data(), y.Data(), {.bias = bias_d.Data()}, stream_.Get());
            ExpectClose(Download(y, stream_), Reference(x, w, 1, shape.in, shape.out, bias, {}), 1e-5);

            DeviceBuffer<float> act(shape.out / 2);
            MatVec(m.View(), x_d.Data(), act.Data(), {.swiglu = true}, stream_.Get());
            ExpectClose(Download(act, stream_), SwiGlu(Reference(x, w, 1, shape.in, shape.out, {}, {})), 1e-5);
        }
    }
}

// x rounded to FP16, as GemmF16 receives it, and the raw bits to upload.
struct HalfInput {
    std::vector<std::uint16_t> bits;
    std::vector<float> values;
};

HalfInput RoundToHalf(const std::vector<float>& x) {
    HalfInput h;
    for (const float v : x) {
        const __half rounded = __float2half_rn(v);
        h.bits.push_back(__half_as_ushort(rounded));
        h.values.push_back(__half2float(rounded));
    }
    return h;
}

TEST_F(QuantizationTest, GemmF16MatchesTheReferenceForEveryFormatAndShape) {
    // Row counts cover each block height (16, 32, 64), partial and multiple
    // row tiles; the shapes cover split K (few output tiles), one split
    // (many), and an output count that isn't a multiple of the tile width.
    // INT4 weights are reconstructed in FP16 inside the kernel (11
    // significant bits), so results agree to about 1e-3 of the largest output.
    struct Shape {
        int out;
        int in;
    };
    GemmWorkspace workspace;
    for (const WeightFormat format : kFormats) {
        for (const Shape shape : {Shape{256, 1536}, Shape{300, 8960}, Shape{8192, 1536}}) {
            const WeightMatrix m = MatrixFrom(RandomWeights(shape.out, shape.in, 7), shape.out, shape.in, format);
            const std::vector<float> w = Decoded(m, stream_);
            const auto bias = RandomFloats(shape.out, 8);
            const auto bias_d = Upload(bias);
            for (const int rows : {1, 2, 16, 17, 40, 64, 100}) {
                SCOPED_TRACE(std::string(WeightFormatName(format)) + " " + std::to_string(rows) + " rows x " +
                             std::to_string(shape.out) + "x" + std::to_string(shape.in));
                const HalfInput x = RoundToHalf(RandomFloats(static_cast<std::size_t>(rows) * shape.in, 9));
                const auto x_d = Upload(x.bits);
                const std::vector<double> plain = Reference(x.values, w, rows, shape.in, shape.out, {}, {});
                const auto start = RandomFloats(plain.size(), 10);
                std::vector<double> with_bias = plain;
                std::vector<double> accumulated = plain;
                for (std::size_t i = 0; i < plain.size(); ++i) {
                    with_bias[i] += bias[i % shape.out];
                    accumulated[i] += start[i];
                }

                DeviceBuffer<float> y(plain.size());
                GemmF16(x_d.Data(), rows, m.View(), y.Data(), {.bias = bias_d.Data()}, workspace, stream_.Get());
                ExpectClose(Download(y, stream_), with_bias, 1e-3);

                auto acc = Upload(start);
                GemmF16(x_d.Data(), rows, m.View(), acc.Data(), {.accumulate = true}, workspace, stream_.Get());
                ExpectClose(Download(acc, stream_), accumulated, 1e-3);

                DeviceBuffer<float> act(static_cast<std::size_t>(rows) * shape.out / 2);
                GemmF16(x_d.Data(), rows, m.View(), act.Data(), {.swiglu = true}, workspace, stream_.Get());
                ExpectClose(Download(act, stream_), SwiGlu(plain), 1e-3);
            }
        }
    }
}

TEST_F(QuantizationTest, GemmF16IsDeterministic) {
    // Split-K partial sums are added in a fixed order, so repeated runs agree
    // bit for bit (no atomics).
    GemmWorkspace workspace;
    const WeightMatrix m = MatrixFrom(RandomWeights(512, 8960, 11), 512, 8960, WeightFormat::kInt4);
    const HalfInput x = RoundToHalf(RandomFloats(8 * 8960, 12));
    const auto x_d = Upload(x.bits);
    DeviceBuffer<float> y(8 * 512);
    GemmF16(x_d.Data(), 8, m.View(), y.Data(), {}, workspace, stream_.Get());
    const std::vector<float> first = Download(y, stream_);
    for (int run = 0; run < 3; ++run) {
        GemmF16(x_d.Data(), 8, m.View(), y.Data(), {}, workspace, stream_.Get());
        EXPECT_EQ(Download(y, stream_), first);
    }
}

}  // namespace
}  // namespace murmur
