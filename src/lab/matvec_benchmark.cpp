#include "lab/lab.hpp"

#include "common/cuda_check.hpp"
#include "common/device_buffer.hpp"
#include "common/gpu_timer.hpp"
#include "engine/blas.hpp"
#include "engine/kernels/gemm.cuh"
#include "engine/kernels/matvec.cuh"
#include "engine/weight_matrix.hpp"

#include <algorithm>
#include <cstdlib>
#include <format>
#include <functional>
#include <optional>
#include <print>
#include <string>
#include <vector>

namespace murmur::lab {

namespace {

constexpr double kPeakBandwidthGBs = 736.0;
// Short bursts run at idle memory clocks; keep the GPU busy long enough to
// reach full clocks before timing.
constexpr float kWarmupMilliseconds = 300.0f;
// Best of several batches: other GPU work (a desktop compositor, an open game
// editor) can only make a batch slower, so the fastest is the kernel's own speed.
constexpr int kBatches = 15;
constexpr int kRunsPerBatch = 20;

struct Shape {
    const char* name;
    int weight_rows;
    int in_features;
    int per_token;   // how many times one decode step runs this product (28 layers)
    bool norm;       // fused RMSNorm prologue
    bool swiglu;     // fused SwiGLU epilogue
};

// Every product a Qwen2.5-1.5B decode step runs, with the fusions it runs them with.
constexpr Shape kShapes[] = {
    {"norm+qkv", 2048, 1536, 28, true, false},
    {"o_proj+add", 1536, 1536, 28, false, false},
    {"norm+gate/up", 17920, 1536, 28, true, true},
    {"down+add", 1536, 8960, 28, false, false},
    {"norm+lm_head", 151936, 1536, 1, true, false},
};

// Synthetic weights in one format: packed values and the scales the format
// needs (contents don't matter for speed).
struct Weights {
    DeviceBuffer<std::uint8_t> data;
    DeviceBuffer<float> row_scales;
    DeviceBuffer<std::uint32_t> group_ranges;
    WeightView view;

    Weights(WeightFormat format, int rows, int cols) {
        const std::size_t elements = static_cast<std::size_t>(rows) * cols;
        data = DeviceBuffer<std::uint8_t>(format == WeightFormat::kBf16  ? 2 * elements
                                          : format == WeightFormat::kFp8 ? elements
                                                                         : elements / 2);
        CudaCheck(cudaMemset(data.Data(), 0x3C, data.SizeBytes()));
        if (format == WeightFormat::kFp8) {
            row_scales = DeviceBuffer<float>(rows);
            CudaCheck(cudaMemset(row_scales.Data(), 0, row_scales.SizeBytes()));
        } else if (format == WeightFormat::kInt4) {
            group_ranges = DeviceBuffer<std::uint32_t>(elements / kInt4GroupSize);
            CudaCheck(cudaMemset(group_ranges.Data(), 0, group_ranges.SizeBytes()));
        }
        view = {.format = format,
                .out_features = rows,
                .in_features = cols,
                .data = data.Data(),
                .row_scales = row_scales.Data(),
                .group_ranges = group_ranges.Data()};
    }

    std::size_t Bytes() const { return data.SizeBytes() + row_scales.SizeBytes() + group_ranges.SizeBytes(); }
};

// The best time of `run(i)` per call, in microseconds, after warming the
// GPU up to full clocks.
double BestMicroseconds(const std::function<void(int)>& run) {
    GpuTimer timer;
    for (float warm = 0.0f; warm < kWarmupMilliseconds;) {
        timer.Start();
        for (int i = 0; i < 20; ++i) {
            run(i);
        }
        warm += timer.StopMilliseconds();
    }
    double best_ms = 1e30;
    for (int batch = 0; batch < kBatches; ++batch) {
        timer.Start();
        for (int i = 0; i < kRunsPerBatch; ++i) {
            run(i);
        }
        best_ms = std::min<double>(best_ms, timer.StopMilliseconds());
    }
    return 1e3 * best_ms / kRunsPerBatch;
}

}  // namespace

// The multi-row half of `matvec --rows N`: GemmF16 (and cuBLAS for BF16) on
// every product of a decode step for N conversations.
int RunGemmRows(WeightFormat format, int rows) {
    std::println("{} weights, {} rows\n", WeightFormatName(format), rows);
    std::println("{:<16} {:>12} {:>13} {:>8} {:>13} {:>8}", "shape", "out x in", "GemmF16 us", "% peak",
                 format == WeightFormat::kBf16 ? "cuBLAS us" : "", format == WeightFormat::kBf16 ? "% peak" : "");
    GemmWorkspace workspace;
    Blas blas(nullptr);
    double gemm_ms = 0.0;
    double cublas_ms = 0.0;
    double total_bytes = 0.0;
    for (const Shape& shape : kShapes) {
        std::vector<Weights> weights;
        weights.emplace_back(format, shape.weight_rows, shape.in_features);
        const std::size_t bytes = weights.front().Bytes();
        const int copies = std::max<int>(2, static_cast<int>(192'000'000 / bytes) + 1);
        while (static_cast<int>(weights.size()) < copies) {
            weights.emplace_back(format, shape.weight_rows, shape.in_features);
        }
        DeviceBuffer<std::uint16_t> x(static_cast<std::size_t>(rows) * shape.in_features);
        DeviceBuffer<float> y(static_cast<std::size_t>(rows) * shape.weight_rows);
        CudaCheck(cudaMemset(x.Data(), 0, x.SizeBytes()));
        const MatVecOptions options{.swiglu = shape.swiglu};
        const double gemm_us = BestMicroseconds([&](int i) {
            GemmF16(x.Data(), rows, weights[i % copies].view, y.Data(), options, workspace, nullptr);
        });
        double cublas_us = 0.0;
        if (format == WeightFormat::kBf16) {
            cublas_us = BestMicroseconds([&](int i) {
                blas.LinearBf16(x.Data(), reinterpret_cast<const std::uint16_t*>(weights[i % copies].view.data), y.Data(),
                                rows, shape.in_features, shape.weight_rows);
            });
        }
        const auto percent = [&](double us) { return 100.0 * bytes / (us * 1e3) / kPeakBandwidthGBs; };
        gemm_ms += gemm_us * shape.per_token / 1e3;
        cublas_ms += cublas_us * shape.per_token / 1e3;
        total_bytes += static_cast<double>(bytes) * shape.per_token;
        if (format == WeightFormat::kBf16) {
            std::println("{:<16} {:>6} x {:<5} {:>13.1f} {:>7.0f}% {:>13.1f} {:>7.0f}%", shape.name, shape.weight_rows,
                         shape.in_features, gemm_us, percent(gemm_us), cublas_us, percent(cublas_us));
        } else {
            std::println("{:<16} {:>6} x {:<5} {:>13.1f} {:>7.0f}%", shape.name, shape.weight_rows, shape.in_features,
                         gemm_us, percent(gemm_us));
        }
    }
    const double ideal = total_bytes / (kPeakBandwidthGBs * 1e6);
    std::println("\nall weights of one step: GemmF16 {:.2f} ms{}; ideal {:.2f} ms", gemm_ms,
                 format == WeightFormat::kBf16 ? std::format(", cuBLAS {:.2f} ms", cublas_ms) : std::string(), ideal);
    return EXIT_SUCCESS;
}

// Times each fused product of a decode step in isolation and reports the
// best achieved bandwidth (bytes of weights, scales included, read per
// second). The weights are bigger than the 64 MB L2 for all but the
// smallest shapes, and a different matrix is used on every run so none of
// them stays cached. One row runs MatVec; several rows (a decode batch) run
// GemmF16 and, for BF16 weights, cuBLAS's BF16 GEMM next to it.
int RunMatVecBenchmark(Args args) {
    WeightFormat format = WeightFormat::kBf16;
    int rows = 1;
    ArgParser parser("murmur-lab matvec");
    parser.Choice("--weights", "bf16|fp8|int4", format, ParseWeightFormat, "the weight format (default bf16)")
        .PositiveInt("--rows", "<N>", rows, "rows multiplied at once: a decode batch (default 1)");
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }
    if (rows > 1) {
        return RunGemmRows(format, rows);
    }
    std::println("{} weights\n", WeightFormatName(format));
    std::println("{:<16} {:>12} {:>10} {:>9} {:>8} {:>13}", "shape", "out x in", "us/call", "GB/s", "% peak",
                 "ms per token");
    double total_ms = 0.0;
    double total_bytes = 0.0;
    for (const Shape& shape : kShapes) {
        std::vector<Weights> weights;
        weights.emplace_back(format, shape.weight_rows, shape.in_features);
        const std::size_t bytes = weights.front().Bytes();
        // Enough copies to exceed L2 (64 MB) so each run reads from DRAM.
        const int copies = std::max<int>(2, static_cast<int>(192'000'000 / bytes) + 1);
        while (static_cast<int>(weights.size()) < copies) {
            weights.emplace_back(format, shape.weight_rows, shape.in_features);
        }
        DeviceBuffer<float> x(shape.in_features);
        DeviceBuffer<float> norm(shape.in_features);
        DeviceBuffer<float> y(shape.weight_rows);
        CudaCheck(cudaMemset(x.Data(), 0, x.SizeBytes()));
        CudaCheck(cudaMemset(norm.Data(), 0, norm.SizeBytes()));
        const MatVecOptions options{.norm_weight = shape.norm ? norm.Data() : nullptr,
                                    .norm_epsilon = 1e-6f,
                                    .swiglu = shape.swiglu};

        const auto run = [&](int i) {
            MatVec(weights[i % copies].view, x.Data(), y.Data(), options, nullptr);
        };
        GpuTimer timer;
        for (float warm = 0.0f; warm < kWarmupMilliseconds;) {
            timer.Start();
            for (int i = 0; i < 20; ++i) {
                run(i);
            }
            warm += timer.StopMilliseconds();
        }
        double best_ms = 1e30;
        for (int batch = 0; batch < kBatches; ++batch) {
            timer.Start();
            for (int i = 0; i < kRunsPerBatch; ++i) {
                run(i);
            }
            best_ms = std::min<double>(best_ms, timer.StopMilliseconds());
        }
        const double us = 1e3 * best_ms / kRunsPerBatch;
        const double gbs = bytes / (us * 1e3);
        const double ms_per_token = us * shape.per_token / 1e3;
        total_ms += ms_per_token;
        total_bytes += static_cast<double>(bytes) * shape.per_token;
        std::println("{:<16} {:>6} x {:<5} {:>9.1f} {:>9.0f} {:>7.0f}% {:>13.3f}", shape.name, shape.weight_rows,
                     shape.in_features, us, gbs, 100.0 * gbs / kPeakBandwidthGBs, ms_per_token);
    }
    const double total_gbs = total_bytes / (total_ms * 1e6);
    std::println("\nall weights of one token: {:.2f} ms = {:.0f} GB/s ({:.0f}% of peak; ideal {:.2f} ms)", total_ms,
                 total_gbs, 100.0 * total_gbs / kPeakBandwidthGBs, total_bytes / (kPeakBandwidthGBs * 1e6));
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
