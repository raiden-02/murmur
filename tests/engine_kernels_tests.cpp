#include "common/cuda_check.hpp"
#include "common/cuda_stream.hpp"
#include "common/device_buffer.hpp"
#include "engine/blas.hpp"
#include "engine/linear.hpp"
#include "engine/kernels/argmax.cuh"
#include "engine/kernels/attention.cuh"
#include "engine/kernels/decode_attention.cuh"
#include "engine/kernels/elementwise.cuh"
#include "engine/kernels/matvec.cuh"
#include "engine/kernels/quantize.cuh"
#include "engine/kernels/rms_norm.cuh"
#include "engine/kernels/rope.cuh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace murmur {
namespace {

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

class EngineKernelsTest : public ::testing::Test {
protected:
    CudaStream stream_;
};

TEST_F(EngineKernelsTest, ConvertBf16ToF32IsExact) {
    // BF16 patterns of 1.0, -2.5, the largest finite BF16, and +infinity.
    const std::vector<std::uint16_t> bits = {0x3F80, 0xC020, 0x7F7F, 0x7F80};
    const auto in = Upload(bits);
    DeviceBuffer<float> out(bits.size());
    ConvertBf16ToF32(in.Data(), out.Data(), bits.size(), stream_.Get());
    const std::vector<float> result = Download(out, stream_);
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const std::uint32_t expected_bits = static_cast<std::uint32_t>(bits[i]) << 16;
        std::uint32_t actual_bits = 0;
        std::memcpy(&actual_bits, &result[i], sizeof(actual_bits));
        EXPECT_EQ(actual_bits, expected_bits);
    }
}

TEST_F(EngineKernelsTest, DequantizeRowsLooksUpBf16Rows) {
    constexpr int kRows = 10;
    constexpr int kWidth = 37;
    std::vector<std::uint16_t> table(kRows * kWidth);
    for (std::size_t i = 0; i < table.size(); ++i) {
        table[i] = static_cast<std::uint16_t>(0x3F80 + i);   // 1.0 and its neighbours, all distinct
    }
    const std::vector<std::int32_t> ids = {3, 0, 9, 3};
    const auto table_d = Upload(table);
    const auto ids_d = Upload(ids);
    DeviceBuffer<float> out(ids.size() * kWidth);
    const WeightView view{.format = WeightFormat::kBf16,
                          .out_features = kRows,
                          .in_features = kWidth,
                          .data = reinterpret_cast<const std::uint8_t*>(table_d.Data())};
    DequantizeRows(view, ids_d.Data(), static_cast<int>(ids.size()), out.Data(), stream_.Get());
    const std::vector<float> result = Download(out, stream_);
    for (std::size_t t = 0; t < ids.size(); ++t) {
        for (int c = 0; c < kWidth; ++c) {
            const std::uint32_t bits = static_cast<std::uint32_t>(table[ids[t] * kWidth + c]) << 16;
            float expected = 0.0f;
            std::memcpy(&expected, &bits, sizeof(expected));
            EXPECT_EQ(result[t * kWidth + c], expected);
        }
    }
}

TEST_F(EngineKernelsTest, AddBiasAddsToEveryRow) {
    constexpr int kRows = 3;
    constexpr int kWidth = 300;
    const std::vector<float> x = RandomFloats(kRows * kWidth, 2);
    const std::vector<float> bias = RandomFloats(kWidth, 4);
    auto x_d = Upload(x);
    const auto bias_d = Upload(bias);
    AddBias(x_d.Data(), bias_d.Data(), kRows, kWidth, stream_.Get());
    const std::vector<float> result = Download(x_d, stream_);
    for (int i = 0; i < kRows * kWidth; ++i) {
        EXPECT_EQ(result[i], x[i] + bias[i % kWidth]);
    }
}

TEST_F(EngineKernelsTest, SwiGluPairsMatchesFormula) {
    constexpr int kRows = 3;
    constexpr int kWidth = 333;
    const std::vector<float> gate_up = RandomFloats(kRows * 2 * kWidth, 5, 8.0f);
    const auto gate_up_d = Upload(gate_up);
    DeviceBuffer<float> out(kRows * kWidth);
    SwiGluPairs(gate_up_d.Data(), out.Data(), kRows, kWidth, stream_.Get());
    std::vector<double> expected(kRows * kWidth);
    for (int r = 0; r < kRows; ++r) {
        for (int i = 0; i < kWidth; ++i) {
            const double g = gate_up[r * 2 * kWidth + 2 * i];
            expected[r * kWidth + i] = g / (1.0 + std::exp(-g)) * gate_up[r * 2 * kWidth + 2 * i + 1];
        }
    }
    ExpectClose(Download(out, stream_), expected, 1e-6);
}

TEST_F(EngineKernelsTest, RmsNormMatchesFormula) {
    constexpr int kRows = 5;
    constexpr int kWidth = 1536;
    constexpr float kEpsilon = 1e-6f;
    const std::vector<float> x = RandomFloats(kRows * kWidth, 7, 3.0f);
    const std::vector<float> weight = RandomFloats(kWidth, 8);
    const auto x_d = Upload(x);
    const auto w_d = Upload(weight);
    DeviceBuffer<float> out(x.size());
    RmsNorm(x_d.Data(), w_d.Data(), out.Data(), kRows, kWidth, kEpsilon, stream_.Get());
    std::vector<double> expected(x.size());
    for (int r = 0; r < kRows; ++r) {
        double sum = 0.0;
        for (int c = 0; c < kWidth; ++c) {
            sum += static_cast<double>(x[r * kWidth + c]) * x[r * kWidth + c];
        }
        const double inverse_rms = 1.0 / std::sqrt(sum / kWidth + kEpsilon);
        for (int c = 0; c < kWidth; ++c) {
            expected[r * kWidth + c] = weight[c] * x[r * kWidth + c] * inverse_rms;
        }
    }
    ExpectClose(Download(out, stream_), expected, 1e-5);
}

TEST_F(EngineKernelsTest, RopeAndCacheRotatesQueriesAndKeysAndCopiesValues) {
    constexpr int kRows = 4;
    constexpr int kHeads = 3;
    constexpr int kKvHeads = 1;
    constexpr int kHeadDim = 128;
    constexpr int kStart = 1000;   // large positions exercise the argument reduction
    constexpr int kSlots = kHeads + 2 * kKvHeads;
    constexpr int kCacheRows = 40;
    const std::vector<float> frequencies = RopeInverseFrequencies(kHeadDim, 1e6f);
    EXPECT_FLOAT_EQ(frequencies[0], 1.0f);
    const std::vector<float> qkv = RandomFloats(kRows * kSlots * kHeadDim, 9);
    const std::vector<int> positions = {kStart, kStart + 1, kStart + 2, kStart + 3};
    const std::vector<int> cache_rows = {37, 2, 19, 20};   // anywhere in the cache: rows of different pages
    const auto qkv_d = Upload(qkv);
    const auto freq_d = Upload(frequencies);
    const auto positions_d = Upload(positions);
    const auto cache_rows_d = Upload(cache_rows);
    DeviceBuffer<float> q(kRows * kHeads * kHeadDim);
    DeviceBuffer<float> k_cache(kCacheRows * kKvHeads * kHeadDim);
    DeviceBuffer<float> v_cache(kCacheRows * kKvHeads * kHeadDim);
    RopeAndCacheQkv(qkv_d.Data(), kRows, positions_d.Data(), cache_rows_d.Data(), freq_d.Data(),
                    {.num_heads = kHeads, .num_kv_heads = kKvHeads, .head_dim = kHeadDim}, q.Data(),
                    {.format = KvFormat::kFp32, .keys = k_cache.Data(), .values = v_cache.Data()}, stream_.Get());

    // Rotates one head vector the way the reference does, in double.
    const auto rotate = [&](const float* in, int position, std::vector<double>& out) {
        constexpr int kHalf = kHeadDim / 2;
        for (int i = 0; i < kHalf; ++i) {
            const double angle = static_cast<double>(static_cast<float>(position) * frequencies[i]);
            out.push_back(in[i] * std::cos(angle) - in[i + kHalf] * std::sin(angle));
        }
        for (int i = 0; i < kHalf; ++i) {
            const double angle = static_cast<double>(static_cast<float>(position) * frequencies[i]);
            out.push_back(in[i + kHalf] * std::cos(angle) + in[i] * std::sin(angle));
        }
    };
    std::vector<double> expected_q;
    for (int r = 0; r < kRows; ++r) {
        for (int h = 0; h < kHeads; ++h) {
            rotate(qkv.data() + (r * kSlots + h) * kHeadDim, positions[r], expected_q);
        }
    }
    ExpectClose(Download(q, stream_), expected_q, 1e-5);
    const std::vector<float> keys = Download(k_cache, stream_);
    const std::vector<float> values = Download(v_cache, stream_);
    for (int r = 0; r < kRows; ++r) {
        const float* row = qkv.data() + r * kSlots * kHeadDim;
        std::vector<double> expected_k;
        rotate(row + kHeads * kHeadDim, positions[r], expected_k);
        const auto key = keys.begin() + cache_rows[r] * kHeadDim;
        ExpectClose(std::vector<float>(key, key + kHeadDim), expected_k, 1e-5);
        const auto value = values.begin() + cache_rows[r] * kHeadDim;
        EXPECT_EQ(std::vector<float>(value, value + kHeadDim),
                  std::vector<float>(row + (kHeads + 1) * kHeadDim, row + (kHeads + 2) * kHeadDim));
    }
}

// One sequence's attention problem, laid out contiguously: query rows for
// positions start_position.., and the keys and values of every position.
struct LogicalAttention {
    int query_count;
    int start_position;
    int num_heads;
    int num_kv_heads;
    int head_dim;

    int Context() const { return start_position + query_count; }
    AttentionShape Shape() const { return {.num_heads = num_heads, .num_kv_heads = num_kv_heads, .head_dim = head_dim}; }
};

// Plain-loop causal attention with grouped-query heads, in double precision.
std::vector<double> ReferenceAttention(const std::vector<float>& q, const std::vector<float>& k,
                                       const std::vector<float>& v, const LogicalAttention& s) {
    const int q_width = s.num_heads * s.head_dim;
    const int kv_width = s.num_kv_heads * s.head_dim;
    const int group = s.num_heads / s.num_kv_heads;
    std::vector<double> out(static_cast<std::size_t>(s.query_count) * q_width);
    for (int t = 0; t < s.query_count; ++t) {
        const int position = s.start_position + t;
        for (int h = 0; h < s.num_heads; ++h) {
            const int kv = h / group;
            std::vector<double> scores(position + 1);
            double max_score = -std::numeric_limits<double>::infinity();
            for (int j = 0; j <= position; ++j) {
                double dot = 0.0;
                for (int d = 0; d < s.head_dim; ++d) {
                    dot += static_cast<double>(q[t * q_width + h * s.head_dim + d]) * k[j * kv_width + kv * s.head_dim + d];
                }
                scores[j] = dot / std::sqrt(static_cast<double>(s.head_dim));
                max_score = std::max(max_score, scores[j]);
            }
            double sum = 0.0;
            for (double& score : scores) {
                score = std::exp(score - max_score);
                sum += score;
            }
            for (int d = 0; d < s.head_dim; ++d) {
                double acc = 0.0;
                for (int j = 0; j <= position; ++j) {
                    acc += scores[j] / sum * v[j * kv_width + kv * s.head_dim + d];
                }
                out[t * q_width + h * s.head_dim + d] = acc;
            }
        }
    }
    return out;
}

// Scatters each sequence's contiguous keys and values over shuffled pages of
// one shared cache, the way the paged KV cache stores them, and builds the
// page tables (one row per sequence).
struct PagedCache {
    static constexpr int kPageSize = 4;
    static constexpr int kPageShift = 2;

    PagedCache(const std::vector<std::vector<float>>& keys, const std::vector<std::vector<float>>& values,
               int kv_width, unsigned seed) {
        int total_pages = 0;
        for (const auto& k : keys) {
            const int rows = static_cast<int>(k.size()) / kv_width;
            stride = std::max(stride, (rows + kPageSize - 1) / kPageSize);
            total_pages += (rows + kPageSize - 1) / kPageSize;
        }
        total_pages += 3;   // a few pages nobody owns
        std::vector<int> free_pages(total_pages);
        std::iota(free_pages.begin(), free_pages.end(), 0);
        std::shuffle(free_pages.begin(), free_pages.end(), std::mt19937(seed));
        std::vector<float> k_cache(static_cast<std::size_t>(total_pages) * kPageSize * kv_width, -99.0f);
        std::vector<float> v_cache(k_cache.size(), -99.0f);
        tables.assign(keys.size() * stride, -1);
        for (std::size_t s = 0; s < keys.size(); ++s) {
            const int rows = static_cast<int>(keys[s].size()) / kv_width;
            for (int page = 0; page * kPageSize < rows; ++page) {
                tables[s * stride + page] = free_pages.back();
                free_pages.pop_back();
            }
            for (int row = 0; row < rows; ++row) {
                const int physical = tables[s * stride + row / kPageSize] * kPageSize + row % kPageSize;
                std::copy_n(keys[s].begin() + row * kv_width, kv_width, k_cache.begin() + physical * kv_width);
                std::copy_n(values[s].begin() + row * kv_width, kv_width, v_cache.begin() + physical * kv_width);
            }
        }
        k_device = Upload(k_cache);
        v_device = Upload(v_cache);
        tables_device = Upload(tables);
    }

    PagedKv Kv() const {
        return {.page_tables = tables_device.Data(), .page_table_stride = stride, .page_shift = kPageShift};
    }

    KvLayer Layer() const {
        return {.format = KvFormat::kFp32,
                .keys = const_cast<float*>(k_device.Data()),
                .values = const_cast<float*>(v_device.Data())};
    }

    int stride = 1;
    std::vector<int> tables;
    DeviceBuffer<float> k_device;
    DeviceBuffer<float> v_device;
    DeviceBuffer<int> tables_device;
};

// Runs CausalAttention for one sequence whose keys and values sit in
// shuffled pages.
std::vector<float> RunAttention(const std::vector<float>& q, const std::vector<float>& k,
                                const std::vector<float>& v, const LogicalAttention& s, const CudaStream& stream) {
    const PagedCache cache({k}, {v}, s.num_kv_heads * s.head_dim, 77);
    std::vector<int> positions(s.query_count);
    std::iota(positions.begin(), positions.end(), s.start_position);
    const auto positions_d = Upload(positions);
    const auto sequences_d = Upload(std::vector<int>(s.query_count, 0));
    const auto q_d = Upload(q);
    DeviceBuffer<float> out(static_cast<std::size_t>(s.query_count) * s.num_heads * s.head_dim);
    CausalAttention(q_d.Data(), cache.Layer(), cache.Kv(),
                    {.positions = positions_d.Data(), .sequences = sequences_d.Data()}, s.query_count, s.Context(),
                    out.Data(), s.Shape(), stream.Get());
    return Download(out, stream);
}

TEST_F(EngineKernelsTest, AttentionMatchesReferenceForPrefillAndDecodeShapes) {
    const LogicalAttention shapes[] = {
        {.query_count = 7, .start_position = 0, .num_heads = 4, .num_kv_heads = 2, .head_dim = 64},     // prefill
        {.query_count = 1, .start_position = 40, .num_heads = 12, .num_kv_heads = 2, .head_dim = 128},  // decode
        {.query_count = 5, .start_position = 9, .num_heads = 6, .num_kv_heads = 6, .head_dim = 32},     // chunk, no GQA
    };
    unsigned seed = 10;
    for (const LogicalAttention& shape : shapes) {
        const auto q = RandomFloats(static_cast<std::size_t>(shape.query_count) * shape.num_heads * shape.head_dim,
                                    seed++, 2.0f);
        const auto k = RandomFloats(static_cast<std::size_t>(shape.Context()) * shape.num_kv_heads * shape.head_dim,
                                    seed++, 2.0f);
        const auto v = RandomFloats(static_cast<std::size_t>(shape.Context()) * shape.num_kv_heads * shape.head_dim,
                                    seed++);
        ExpectClose(RunAttention(q, k, v, shape, stream_), ReferenceAttention(q, k, v, shape), 1e-5);
    }
}

TEST_F(EngineKernelsTest, AttentionRowsFromDifferentSequencesShareOneLaunch) {
    // Two prompts' chunks in one call: 3 rows of sequence 0 (positions 5..7)
    // and 2 rows of sequence 1 (positions 0..1), each reading its own pages.
    const LogicalAttention a{.query_count = 3, .start_position = 5, .num_heads = 4, .num_kv_heads = 2, .head_dim = 32};
    const LogicalAttention b{.query_count = 2, .start_position = 0, .num_heads = 4, .num_kv_heads = 2, .head_dim = 32};
    const int q_width = a.num_heads * a.head_dim;
    const int kv_width = a.num_kv_heads * a.head_dim;
    const auto qa = RandomFloats(static_cast<std::size_t>(a.query_count) * q_width, 40, 2.0f);
    const auto qb = RandomFloats(static_cast<std::size_t>(b.query_count) * q_width, 41, 2.0f);
    const auto ka = RandomFloats(static_cast<std::size_t>(a.Context()) * kv_width, 42, 2.0f);
    const auto va = RandomFloats(static_cast<std::size_t>(a.Context()) * kv_width, 43);
    const auto kb = RandomFloats(static_cast<std::size_t>(b.Context()) * kv_width, 44, 2.0f);
    const auto vb = RandomFloats(static_cast<std::size_t>(b.Context()) * kv_width, 45);
    const PagedCache cache({ka, kb}, {va, vb}, kv_width, 46);

    std::vector<float> q = qa;
    q.insert(q.end(), qb.begin(), qb.end());
    const auto q_d = Upload(q);
    const auto positions_d = Upload(std::vector<int>{5, 6, 7, 0, 1});
    const auto sequences_d = Upload(std::vector<int>{0, 0, 0, 1, 1});
    DeviceBuffer<float> out(q.size());
    CausalAttention(q_d.Data(), cache.Layer(), cache.Kv(),
                    {.positions = positions_d.Data(), .sequences = sequences_d.Data()}, 5, a.Context(), out.Data(),
                    a.Shape(), stream_.Get());
    const std::vector<float> result = Download(out, stream_);
    ExpectClose(std::vector<float>(result.begin(), result.begin() + qa.size()), ReferenceAttention(qa, ka, va, a), 1e-5);
    ExpectClose(std::vector<float>(result.begin() + qa.size(), result.end()), ReferenceAttention(qb, kb, vb, b), 1e-5);
}

// FlashAttention against the double-precision reference: three prompts in
// one call, two of them continuing after keys already cached, their rows
// crossing the kernel's 64-row tiles (rows 64 and 128 fall inside the first
// two prompts), for both head sizes the models use. FP16 queries, keys and
// values keep about 11 significant bits.
TEST_F(EngineKernelsTest, FlashAttentionMatchesTheReferenceAcrossTilesAndPrompts) {
    for (const auto [heads, kv_heads, head_dim] : {std::tuple{12, 2, 128}, std::tuple{14, 2, 64}}) {
        SCOPED_TRACE(head_dim);
        const LogicalAttention prompts[] = {
            {.query_count = 100, .start_position = 30, .num_heads = heads, .num_kv_heads = kv_heads, .head_dim = head_dim},
            {.query_count = 70, .start_position = 0, .num_heads = heads, .num_kv_heads = kv_heads, .head_dim = head_dim},
            {.query_count = 5, .start_position = 200, .num_heads = heads, .num_kv_heads = kv_heads, .head_dim = head_dim},
        };
        const int q_width = heads * head_dim;
        const int kv_width = kv_heads * head_dim;
        std::vector<std::vector<float>> qs, ks, vs;
        std::vector<float> q;
        std::vector<int> positions, sequences;
        unsigned seed = 300;
        for (int p = 0; p < 3; ++p) {
            const LogicalAttention& a = prompts[p];
            qs.push_back(RandomFloats(static_cast<std::size_t>(a.query_count) * q_width, seed++, 2.0f));
            ks.push_back(RandomFloats(static_cast<std::size_t>(a.Context()) * kv_width, seed++, 2.0f));
            vs.push_back(RandomFloats(static_cast<std::size_t>(a.Context()) * kv_width, seed++));
            q.insert(q.end(), qs.back().begin(), qs.back().end());
            for (int r = 0; r < a.query_count; ++r) {
                positions.push_back(a.start_position + r);
                sequences.push_back(p);
            }
        }
        const PagedCache cache(ks, vs, kv_width, 301);
        const auto q_d = Upload(q);
        const auto positions_d = Upload(positions);
        const auto sequences_d = Upload(sequences);
        DeviceBuffer<float> out(q.size());
        FlashAttention(q_d.Data(), cache.Layer(), cache.Kv(),
                       {.positions = positions_d.Data(), .sequences = sequences_d.Data()},
                       static_cast<int>(positions.size()), out.Data(), prompts[0].Shape(), stream_.Get());
        const std::vector<float> result = Download(out, stream_);
        std::size_t offset = 0;
        for (int p = 0; p < 3; ++p) {
            SCOPED_TRACE(p);
            const std::size_t size = qs[p].size();
            ExpectClose(std::vector<float>(result.begin() + offset, result.begin() + offset + size),
                        ReferenceAttention(qs[p], ks[p], vs[p], prompts[p]), 2e-3);
            offset += size;
        }
    }
}

TEST_F(EngineKernelsTest, DecodingOneTokenEqualsTheLastRowOfPrefill) {
    // The KV cache contract: attending with start_position = n - 1 and one
    // query must give exactly what a full prefill gives for its last token.
    const LogicalAttention prefill{.query_count = 9, .start_position = 0, .num_heads = 4, .num_kv_heads = 1, .head_dim = 64};
    const int width = prefill.num_heads * prefill.head_dim;
    const auto q = RandomFloats(static_cast<std::size_t>(prefill.query_count) * width, 20, 2.0f);
    const auto k = RandomFloats(static_cast<std::size_t>(prefill.query_count) * prefill.head_dim, 21, 2.0f);
    const auto v = RandomFloats(static_cast<std::size_t>(prefill.query_count) * prefill.head_dim, 22);
    const std::vector<float> full = RunAttention(q, k, v, prefill, stream_);

    LogicalAttention decode = prefill;
    decode.query_count = 1;
    decode.start_position = prefill.query_count - 1;
    const std::vector<float> last_q(q.end() - width, q.end());
    const std::vector<float> single = RunAttention(last_q, k, v, decode, stream_);
    EXPECT_EQ(single, std::vector<float>(full.end() - width, full.end()));
}

// The model's head layout: 12 query heads sharing 2 KV heads (groups of 6).
constexpr int kModelHeads = 12;
constexpr int kModelKvHeads = 2;
constexpr int kModelHeadDim = 128;

TEST_F(EngineKernelsTest, DecodeAttentionMatchesReferenceAtEveryContextLength) {
    // Lengths straddle the split boundaries (16 keys per split up to 2,048
    // keys, more beyond) and reach the full window.
    constexpr int kMaxContext = 4096;
    constexpr int kKvWidth = kModelKvHeads * kModelHeadDim;
    DecodeAttentionWorkspace workspace({.num_heads = kModelHeads, .num_kv_heads = kModelKvHeads,
                                        .head_dim = kModelHeadDim, .max_context = kMaxContext, .max_batch = 1});
    const auto k = RandomFloats(static_cast<std::size_t>(kMaxContext) * kKvWidth, 90, 2.0f);
    const auto v = RandomFloats(static_cast<std::size_t>(kMaxContext) * kKvWidth, 91);
    const auto q = RandomFloats(kModelHeads * kModelHeadDim, 92, 2.0f);
    const PagedCache cache({k}, {v}, kKvWidth, 93);
    const auto q_d = Upload(q);
    DeviceBuffer<int> context_d(1);
    DeviceBuffer<float> out(kModelHeads * kModelHeadDim);
    for (const int context : {1, 15, 16, 17, 100, 2048, 2049, 4096}) {
        context_d.CopyFromHost(std::vector<int>{context});
        DecodeAttention(q_d.Data(), cache.Layer(), cache.Kv(), context_d.Data(), 1,
                        out.Data(), workspace, stream_.Get());
        const LogicalAttention shape{.query_count = 1, .start_position = context - 1, .num_heads = kModelHeads,
                                     .num_kv_heads = kModelKvHeads, .head_dim = kModelHeadDim};
        SCOPED_TRACE("context " + std::to_string(context));
        ExpectClose(Download(out, stream_), ReferenceAttention(q, k, v, shape), 1e-5);
    }
}

TEST_F(EngineKernelsTest, BatchedDecodeAttentionKeepsEverySequenceSeparate) {
    // Three sequences of different lengths decode together, each through its
    // own shuffled pages; each must get exactly its own attention.
    constexpr int kKvWidth = kModelKvHeads * kModelHeadDim;
    constexpr int kQWidth = kModelHeads * kModelHeadDim;
    const std::vector<int> contexts = {33, 1, 700};
    DecodeAttentionWorkspace workspace({.num_heads = kModelHeads, .num_kv_heads = kModelKvHeads,
                                        .head_dim = kModelHeadDim, .max_context = 1024, .max_batch = 4});
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
    std::vector<float> q;
    unsigned seed = 60;
    for (const int context : contexts) {
        keys.push_back(RandomFloats(static_cast<std::size_t>(context) * kKvWidth, seed++, 2.0f));
        values.push_back(RandomFloats(static_cast<std::size_t>(context) * kKvWidth, seed++));
        const auto q_row = RandomFloats(kQWidth, seed++, 2.0f);
        q.insert(q.end(), q_row.begin(), q_row.end());
    }
    const PagedCache cache(keys, values, kKvWidth, 64);
    const auto q_d = Upload(q);
    const auto contexts_d = Upload(contexts);
    DeviceBuffer<float> out(q.size());
    DecodeAttention(q_d.Data(), cache.Layer(), cache.Kv(), contexts_d.Data(),
                    static_cast<int>(contexts.size()), out.Data(), workspace, stream_.Get());
    const std::vector<float> result = Download(out, stream_);
    for (std::size_t s = 0; s < contexts.size(); ++s) {
        const LogicalAttention shape{.query_count = 1, .start_position = contexts[s] - 1, .num_heads = kModelHeads,
                                     .num_kv_heads = kModelKvHeads, .head_dim = kModelHeadDim};
        const std::vector<float> q_row(q.begin() + s * kQWidth, q.begin() + (s + 1) * kQWidth);
        SCOPED_TRACE("sequence " + std::to_string(s));
        ExpectClose(std::vector<float>(result.begin() + s * kQWidth, result.begin() + (s + 1) * kQWidth),
                    ReferenceAttention(q_row, keys[s], values[s], shape), 1e-5);
    }
}

TEST_F(EngineKernelsTest, EveryKvFormatRoundTripsThroughRopeAndAttention) {
    // RoPE writes a sequence's keys and values into a cache in each format;
    // decode attention reads them back. The output must match the FP32
    // cache's to the format's precision: BF16 keeps 8 significant bits, FP8
    // (E4M3, scaled per token and head) 4.
    constexpr int kHeads = 12;
    constexpr int kKvHeads = 2;
    constexpr int kHeadDim = 128;
    constexpr int kContext = 300;
    constexpr int kSlots = kHeads + 2 * kKvHeads;
    constexpr int kKvWidth = kKvHeads * kHeadDim;
    const std::vector<float> frequencies = RopeInverseFrequencies(kHeadDim, 1e6f);
    // Keys larger than values, as in the model; large enough to make the
    // softmax uneven, not so large that one key takes all the weight (then
    // any rounding of the scores flips which one, in any format).
    std::vector<float> qkv = RandomFloats(static_cast<std::size_t>(kContext) * kSlots * kHeadDim, 90);
    for (int r = 0; r < kContext; ++r) {
        for (int i = kHeads * kHeadDim; i < (kHeads + kKvHeads) * kHeadDim; ++i) {
            qkv[static_cast<std::size_t>(r) * kSlots * kHeadDim + i] *= 4.0f;
        }
    }
    std::vector<int> positions(kContext);
    std::iota(positions.begin(), positions.end(), 0);
    const auto qkv_d = Upload(qkv);
    const auto freq_d = Upload(frequencies);
    const auto positions_d = Upload(positions);   // cache row = position: one sequence on pages 0, 1, ...
    const std::vector<int> table(kContext / 4 + 1, 0);
    std::vector<int> pages(table.size());
    std::iota(pages.begin(), pages.end(), 0);
    const auto pages_d = Upload(pages);
    const auto context_d = Upload(std::vector<int>{kContext});
    const PagedKv kv{.page_tables = pages_d.Data(), .page_table_stride = static_cast<int>(pages.size()), .page_shift = 2};
    DecodeAttentionWorkspace workspace(
        {.num_heads = kHeads, .num_kv_heads = kKvHeads, .head_dim = kHeadDim, .max_context = 512, .max_batch = 1});

    std::vector<float> fp32_out;
    for (const auto [format, tolerance] : {std::pair{KvFormat::kFp32, 0.0}, std::pair{KvFormat::kBf16, 2e-2},
                                           std::pair{KvFormat::kFp8, 8e-2}}) {
        const std::size_t bytes = format == KvFormat::kFp32 ? 4 : format == KvFormat::kBf16 ? 2 : 1;
        const std::size_t rows = pages.size() * 4;
        DeviceBuffer<std::uint8_t> keys(rows * kKvWidth * bytes);
        DeviceBuffer<std::uint8_t> values(rows * kKvWidth * bytes);
        DeviceBuffer<float> key_scales(rows * kKvHeads);
        DeviceBuffer<float> value_scales(rows * kKvHeads);
        const KvLayer cache{.format = format,
                            .keys = keys.Data(),
                            .values = values.Data(),
                            .key_scales = key_scales.Data(),
                            .value_scales = value_scales.Data()};
        DeviceBuffer<float> q(static_cast<std::size_t>(kContext) * kHeads * kHeadDim);
        RopeAndCacheQkv(qkv_d.Data(), kContext, positions_d.Data(), positions_d.Data(), freq_d.Data(),
                        {.num_heads = kHeads, .num_kv_heads = kKvHeads, .head_dim = kHeadDim}, q.Data(), cache,
                        stream_.Get());
        // The last token's query attends to all kContext cached positions.
        DeviceBuffer<float> out(kHeads * kHeadDim);
        DecodeAttention(q.Data() + static_cast<std::size_t>(kContext - 1) * kHeads * kHeadDim, cache, kv,
                        context_d.Data(), 1, out.Data(), workspace, stream_.Get());
        const std::vector<float> result = Download(out, stream_);
        if (format == KvFormat::kFp32) {
            fp32_out = result;
            continue;
        }
        SCOPED_TRACE(format == KvFormat::kBf16 ? "bf16" : "fp8");
        ExpectClose(result, std::vector<double>(fp32_out.begin(), fp32_out.end()), tolerance);
    }
}

TEST_F(EngineKernelsTest, AttentionRejectsContextsThatDontFit) {
    EXPECT_THROW(CausalAttention(nullptr, KvLayer{}, PagedKv{}, QueryRows{}, 1, MaxAttentionContext(128) + 1,
                                 nullptr, {.num_heads = 2, .num_kv_heads = 1, .head_dim = 128}, stream_.Get()),
                 std::invalid_argument);
    DecodeAttentionWorkspace workspace({.num_heads = 2, .num_kv_heads = 1, .head_dim = 64, .max_context = 64,
                                        .max_batch = 2});
    EXPECT_THROW(DecodeAttention(nullptr, KvLayer{}, PagedKv{}, nullptr, 3, nullptr, workspace, stream_.Get()),
                 std::invalid_argument);
}

TEST_F(EngineKernelsTest, ArgmaxFindsTheFirstMaximum) {
    std::vector<float> values = RandomFloats(151665, 30);
    values[1234] = 50.0f;
    values[99999] = 50.0f;   // tie: the lower index must win
    values[5] = std::numeric_limits<float>::quiet_NaN();
    const auto values_d = Upload(values);
    DeviceBuffer<std::int32_t> index(1);
    Argmax(values_d.Data(), static_cast<int>(values.size()), index.Data(), stream_.Get());
    EXPECT_EQ(Download(index, stream_)[0], 1234);
}

TEST_F(EngineKernelsTest, LinearMatchesMatrixProduct) {
    constexpr int kRows = 3;
    constexpr int kIn = 70;
    constexpr int kOut = 50;
    const auto x = RandomFloats(kRows * kIn, 40);
    const auto w = RandomFloats(kOut * kIn, 41);
    const auto x_d = Upload(x);
    const auto w_d = Upload(w);
    DeviceBuffer<float> y(kRows * kOut);
    Blas blas(stream_.Get());
    blas.Linear(x_d.Data(), w_d.Data(), y.Data(), kRows, kIn, kOut);
    std::vector<double> expected(kRows * kOut);
    for (int r = 0; r < kRows; ++r) {
        for (int o = 0; o < kOut; ++o) {
            double acc = 0.0;
            for (int i = 0; i < kIn; ++i) {
                acc += static_cast<double>(x[r * kIn + i]) * w[o * kIn + i];
            }
            expected[r * kOut + o] = acc;
        }
    }
    ExpectClose(Download(y, stream_), expected, 1e-5);
}

// Random BF16 weights (the top 16 bits of random floats) and their exact FP32 values.
struct Bf16Weights {
    std::vector<Bf16Bits> bits;
    std::vector<float> values;
};

Bf16Weights RandomBf16(std::size_t count, unsigned seed) {
    Bf16Weights w;
    for (const float f : RandomFloats(count, seed)) {
        std::uint32_t full = 0;
        std::memcpy(&full, &f, sizeof(full));
        const std::uint32_t truncated = full & 0xFFFF0000u;
        float value = 0.0f;
        std::memcpy(&value, &truncated, sizeof(value));
        w.bits.push_back(static_cast<Bf16Bits>(full >> 16));
        w.values.push_back(value);
    }
    return w;
}

WeightView Bf16View(const DeviceBuffer<Bf16Bits>& bits, int out, int in) {
    return {.format = WeightFormat::kBf16,
            .out_features = out,
            .in_features = in,
            .data = reinterpret_cast<const std::uint8_t*>(bits.Data())};
}

// A WeightMatrix in `format` made from BF16 weights.
WeightMatrix MatrixFrom(const Bf16Weights& w, int out, int in, WeightFormat format) {
    DeviceBuffer<std::uint8_t> bytes(w.bits.size() * sizeof(Bf16Bits));
    CudaCheck(cudaMemcpy(bytes.Data(), w.bits.data(), bytes.Size(), cudaMemcpyHostToDevice));
    return WeightMatrix::FromBf16(std::move(bytes), out, in, format);
}

// y[r][o] = start[r][o] + sum_i x[r][i] * W[o][i] + bias[o], in double.
std::vector<double> ReferenceLinear(const std::vector<float>& x, const std::vector<float>& w, int rows, int in,
                                    int out, const std::vector<float>& bias, const std::vector<float>& start) {
    std::vector<double> y(static_cast<std::size_t>(rows) * out);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < out; ++o) {
            double acc = start.empty() ? 0.0 : start[r * out + o];
            for (int i = 0; i < in; ++i) {
                acc += static_cast<double>(x[r * in + i]) * w[static_cast<std::size_t>(o) * in + i];
            }
            y[r * out + o] = acc + (bias.empty() ? 0.0 : bias[o]);
        }
    }
    return y;
}

TEST_F(EngineKernelsTest, MatVecMatchesReferenceOnModelAndOddShapes) {
    struct Shape {
        int out;
        int in;
    };
    // 1536 and 8960 are the model's widths (vector path); 37 exercises the
    // scalar path and 13 rows leave a partly filled block.
    for (const Shape shape : {Shape{256, 1536}, Shape{1536, 8960}, Shape{13, 37}}) {
        const Bf16Weights w = RandomBf16(static_cast<std::size_t>(shape.out) * shape.in, 50);
        const auto x = RandomFloats(shape.in, 51);
        const auto bias = RandomFloats(shape.out, 52);
        const auto start = RandomFloats(shape.out, 53);
        const auto w_d = Upload(w.bits);
        const auto x_d = Upload(x);
        const auto bias_d = Upload(bias);

        DeviceBuffer<float> y(shape.out);
        MatVec(Bf16View(w_d, shape.out, shape.in), x_d.Data(), y.Data(), {.bias = bias_d.Data()}, stream_.Get());
        ExpectClose(Download(y, stream_), ReferenceLinear(x, w.values, 1, shape.in, shape.out, bias, {}), 1e-5);

        auto acc = Upload(start);
        MatVec(Bf16View(w_d, shape.out, shape.in), x_d.Data(), acc.Data(), {.accumulate = true}, stream_.Get());
        ExpectClose(Download(acc, stream_), ReferenceLinear(x, w.values, 1, shape.in, shape.out, {}, start), 1e-5);
    }
}

// RMSNorm of x, the way the fused prologue applies it (in double, rounded).
std::vector<float> ReferenceNorm(const std::vector<float>& x, const std::vector<float>& weight, float epsilon) {
    double sum = 0.0;
    for (const float v : x) {
        sum += static_cast<double>(v) * v;
    }
    const double inverse_rms = 1.0 / std::sqrt(sum / x.size() + epsilon);
    std::vector<float> out;
    for (std::size_t i = 0; i < x.size(); ++i) {
        out.push_back(static_cast<float>(weight[i] * (x[i] * inverse_rms)));
    }
    return out;
}

TEST_F(EngineKernelsTest, MatVecFusesNormAndSwiGlu) {
    constexpr int kIntermediate = 700;
    constexpr int kIn = 1536;
    constexpr float kEpsilon = 1e-6f;
    const Bf16Weights w = RandomBf16(static_cast<std::size_t>(2 * kIntermediate) * kIn, 70);
    const auto x = RandomFloats(kIn, 71, 3.0f);
    const auto norm = RandomFloats(kIn, 72);
    const auto w_d = Upload(w.bits);
    const auto x_d = Upload(x);
    const auto norm_d = Upload(norm);
    const std::vector<float> normed = ReferenceNorm(x, norm, kEpsilon);

    // RMSNorm prologue on a plain product.
    DeviceBuffer<float> y(2 * kIntermediate);
    MatVec(Bf16View(w_d, 2 * kIntermediate, kIn), x_d.Data(), y.Data(),
           {.norm_weight = norm_d.Data(), .norm_epsilon = kEpsilon}, stream_.Get());
    const std::vector<double> gate_up = ReferenceLinear(normed, w.values, 1, kIn, 2 * kIntermediate, {}, {});
    ExpectClose(Download(y, stream_), gate_up, 1e-5);

    // The same with the SwiGLU epilogue: SiLU(gate) * up, rows interleaved.
    DeviceBuffer<float> act(kIntermediate);
    MatVec(Bf16View(w_d, 2 * kIntermediate, kIn), x_d.Data(), act.Data(),
           {.norm_weight = norm_d.Data(), .norm_epsilon = kEpsilon, .swiglu = true}, stream_.Get());
    std::vector<double> expected(kIntermediate);
    for (int i = 0; i < kIntermediate; ++i) {
        expected[i] = gate_up[2 * i] / (1.0 + std::exp(-gate_up[2 * i])) * gate_up[2 * i + 1];
    }
    ExpectClose(Download(act, stream_), expected, 1e-5);
}

TEST_F(EngineKernelsTest, LinearGemmPathMatchesTheFusedKernel) {
    // Several rows take a matrix-product path; it must compute the same
    // function as the fused one-row kernel, for every option and weight
    // format (both read the same quantized weights). The exact path agrees
    // to FP32 rounding; the tensor-core paths round x to FP16 (11 significant
    // bits), BF16 (8) or FP8 (4) first, so they agree to about 2e-3, 1e-2
    // and 3e-2.
    constexpr int kRows = 5;
    constexpr int kIn = 1536;
    constexpr int kOut = 600;
    constexpr float kEpsilon = 1e-6f;
    const Bf16Weights w = RandomBf16(static_cast<std::size_t>(kOut) * kIn, 80);
    const auto x = RandomFloats(kRows * kIn, 81, 3.0f);
    const auto norm = RandomFloats(kIn, 82);
    const auto bias = RandomFloats(kOut, 83);
    const auto x_d = Upload(x);
    const auto norm_d = Upload(norm);
    const auto bias_d = Upload(bias);
    struct Case {
        WeightFormat format;
        MultiRowMath math;
        double tolerance;
    };
    for (const Case c : {Case{WeightFormat::kBf16, MultiRowMath::kExactFp32, 1e-5},
                         Case{WeightFormat::kBf16, MultiRowMath::kCublasBf16, 1e-2},
                         Case{WeightFormat::kBf16, MultiRowMath::kTensorCores, 2e-3},
                         Case{WeightFormat::kFp8, MultiRowMath::kExactFp32, 1e-5},
                         Case{WeightFormat::kFp8, MultiRowMath::kTensorCores, 2e-3},
                         Case{WeightFormat::kFp8, MultiRowMath::kFp8TensorCores, 3e-2},
                         Case{WeightFormat::kBf16, MultiRowMath::kFastest, 1e-2},
                         Case{WeightFormat::kFp8, MultiRowMath::kFastest, 3e-2},
                         Case{WeightFormat::kInt4, MultiRowMath::kFastest, 2e-3},
                         Case{WeightFormat::kInt4, MultiRowMath::kExactFp32, 1e-5},
                         Case{WeightFormat::kInt4, MultiRowMath::kTensorCores, 2e-3}}) {
        SCOPED_TRACE(std::string(WeightFormatName(c.format)) + " math " + std::to_string(static_cast<int>(c.math)));
        const WeightMatrix matrix = MatrixFrom(w, kOut, kIn, c.format);
        const double tolerance = c.tolerance;
        Linear linear(stream_.Get(),
                      {.max_rows = kRows, .max_in_features = kIn, .max_norm_width = kIn, .max_swiglu_rows = kOut,
                       .max_matrix_elements = matrix.ElementCount()},
                      c.math);
        for (const MatVecOptions& options :
             {MatVecOptions{.norm_weight = norm_d.Data(), .norm_epsilon = kEpsilon, .bias = bias_d.Data()},
              MatVecOptions{.norm_weight = norm_d.Data(), .norm_epsilon = kEpsilon, .swiglu = true},
              MatVecOptions{.accumulate = true}}) {
            const int out_width = options.swiglu ? kOut / 2 : kOut;
            DeviceBuffer<float> all_rows(static_cast<std::size_t>(kRows) * out_width);
            CudaCheck(cudaMemset(all_rows.Data(), 0, all_rows.SizeBytes()));
            linear.Apply(x_d.Data(), matrix, all_rows.Data(), kRows, options);

            std::vector<double> one_at_a_time;
            for (int r = 0; r < kRows; ++r) {
                DeviceBuffer<float> y(out_width);
                CudaCheck(cudaMemset(y.Data(), 0, y.SizeBytes()));
                linear.Apply(x_d.Data() + static_cast<std::size_t>(r) * kIn, matrix, y.Data(), 1, options);
                for (const float v : Download(y, stream_)) {
                    one_at_a_time.push_back(v);
                }
            }
            ExpectClose(Download(all_rows, stream_), one_at_a_time, tolerance);
        }
    }
}

TEST_F(EngineKernelsTest, LinearUsesGemmForSeveralRowsWithTheSameResult) {
    constexpr int kRows = 5;
    constexpr int kIn = 1536;
    constexpr int kOut = 300;
    const Bf16Weights w = RandomBf16(static_cast<std::size_t>(kOut) * kIn, 60);
    const auto x = RandomFloats(kRows * kIn, 61);
    const auto bias = RandomFloats(kOut, 62);
    const WeightMatrix matrix = MatrixFrom(w, kOut, kIn, WeightFormat::kBf16);
    const auto x_d = Upload(x);
    const auto bias_d = Upload(bias);
    DeviceBuffer<float> y(kRows * kOut);
    Linear linear(stream_.Get(), {.max_rows = kRows, .max_in_features = kIn, .max_matrix_elements = matrix.ElementCount()},
                  MultiRowMath::kExactFp32);
    linear.Apply(x_d.Data(), matrix, y.Data(), kRows, {.bias = bias_d.Data()});
    ExpectClose(Download(y, stream_), ReferenceLinear(x, w.values, kRows, kIn, kOut, bias, {}), 1e-5);
}

}  // namespace
}  // namespace murmur
