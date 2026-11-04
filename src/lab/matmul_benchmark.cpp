#include "lab/lab.hpp"

#include "common/cuda_check.hpp"
#include "common/device_buffer.hpp"
#include "common/gpu_timer.hpp"
#include "kernels/matmul.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <optional>
#include <print>
#include <random>
#include <vector>

namespace murmur::lab {

namespace {

constexpr int kSize = 4096;
constexpr int kWarmupRuns = 2;
constexpr int kTimedRuns = 10;

// Ada has 128 FP32 units per SM, each able to do one fused multiply-add
// (2 floating-point operations) per clock tick.
constexpr int kFp32UnitsPerSm = 128;

double PeakTflops() {
    int sms = 0;
    int clock_khz = 0;
    CudaCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CudaCheck(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    return static_cast<double>(sms) * kFp32UnitsPerSm * 2.0 * clock_khz * 1e3 / 1e12;
}

float MedianMs(const std::function<void()>& run, GpuTimer& timer) {
    for (int i = 0; i < kWarmupRuns; ++i) run();
    std::vector<float> ms;
    for (int i = 0; i < kTimedRuns; ++i) {
        timer.Start();
        run();
        ms.push_back(timer.StopMilliseconds());
    }
    std::ranges::sort(ms);
    return ms[ms.size() / 2];
}

}  // namespace

// Times C = A x B for 4096 x 4096 matrices with each kernel and with cuBLAS,
// and reports TFLOP/s: a matrix multiply does 2 x m x n x k floating-point
// operations (one multiply and one add per term). Each kernel's result is
// compared with cuBLAS before its time is reported.
int RunMatmulBenchmark(Args args) {
    if (const std::optional<int> exit_code = ArgParser("murmur-lab matmul").Parse(args)) {
        return *exit_code;
    }
    const std::size_t count = static_cast<std::size_t>(kSize) * kSize;
    std::mt19937 random(11);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> host_a(count), host_b(count);
    for (float& v : host_a) v = distribution(random);
    for (float& v : host_b) v = distribution(random);

    DeviceBuffer<float> a(count), b(count), c(count), reference(count);
    a.CopyFromHost(host_a);
    b.CopyFromHost(host_b);

    CublasMatmul cublas;
    GpuTimer timer;
    const double flops = 2.0 * kSize * kSize * static_cast<double>(kSize);
    const double peak = PeakTflops();
    std::println("matmul: {0} x {0} x {0}, {1:.0f} GFLOP per run, FP32 peak ~{2:.1f} TFLOP/s, median of {3}",
                 kSize, flops / 1e9, peak, kTimedRuns);

    const float cublas_ms = MedianMs([&] { cublas.Run(a, b, reference, kSize, kSize, kSize); }, timer);
    std::vector<float> host_reference(count), host_c(count);
    reference.CopyToHost(host_reference);

    const auto report = [&](const char* name, float ms) {
        const double tflops = flops / (ms * 1e-3) / 1e12;
        std::println("  {:<24} {:>8.2f} ms  {:>6.2f} TFLOP/s  ({:.0f}% of peak)", name, ms, tflops,
                     tflops / peak * 100.0);
    };

    for (MatmulVersion version : { MatmulVersion::kNaive, MatmulVersion::kTiled }) {
        const float ms = MedianMs([&] { Matmul(a, b, c, kSize, kSize, kSize, version); }, timer);
        c.CopyToHost(host_c);
        float worst = 0.0f;
        for (std::size_t i = 0; i < count; ++i) {
            worst = std::max(worst, std::abs(host_c[i] - host_reference[i]));
        }
        if (worst > 1e-2f) {
            std::println("  {:<24} WRONG RESULT (max difference from cuBLAS {:.4f})",
                         MatmulVersionName(version), worst);
            continue;
        }
        report(MatmulVersionName(version), ms);
    }
    report("cuBLAS (reference)", cublas_ms);
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
