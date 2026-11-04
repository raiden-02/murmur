#pragma once

#include "common/device_buffer.hpp"
#include "engine/kernels/matvec.cuh"

#include <cuda_runtime.h>

#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace murmur {

class WeightMatrix;

// How strongly each input channel of each weight matrix is driven: the mean
// of x_c^2 over every row that went through the matrix on calibration text.
// Rounding weight (o, c) by e moves output o by x_c * e, so quantization
// minimizes the error weighted by these values (an "importance matrix", as
// llama.cpp calls it). Matrices are named by role: "layers.3.qkv",
// "layers.3.o_proj", "layers.3.gate_up", "layers.3.down", "lm_head".
class ChannelImportance {
public:
    void Set(std::string name, std::vector<float> values);
    const std::vector<float>* Find(std::string_view name) const;
    bool Empty() const { return values_.empty(); }

    void Save(const std::filesystem::path& path) const;
    static ChannelImportance Load(const std::filesystem::path& path);

private:
    std::map<std::string, std::vector<float>, std::less<>> values_;
};

std::filesystem::path ImportancePath(const std::filesystem::path& model_dir);

// Accumulates, on the GPU, the sum of x_c^2 for every input channel of every
// matrix a forward pass multiplies (after the fused RMSNorm, so it sees
// exactly what the matrix sees). Linear calls Add while recording is on.
class InputStatistics {
public:
    void Add(const WeightMatrix& weight, const float* x, int rows, const MatVecOptions& options, cudaStream_t stream);
    std::vector<float> MeanSquares(const WeightMatrix& weight) const;

private:
    struct Sums {
        DeviceBuffer<float> squares;
        long long rows = 0;
    };
    std::unordered_map<const WeightMatrix*, Sums> sums_;
};

}  // namespace murmur
