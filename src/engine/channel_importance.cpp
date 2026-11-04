#include "engine/channel_importance.hpp"

#include "common/cuda_check.hpp"
#include "engine/kernels/quantize.cuh"
#include "engine/weight_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace murmur {

namespace {

constexpr char kMagic[8] = {'M', 'U', 'R', 'I', 'M', 'P', '0', '1'};
constexpr std::uint32_t kMaxNameBytes = 256;   // tensor names are ~40 bytes

template <typename T>
void Write(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
T Read(std::ifstream& in) {
    T value{};
    if (!in.read(reinterpret_cast<char*>(&value), sizeof(value))) {
        throw std::runtime_error("importance file: truncated");
    }
    return value;
}

}  // namespace

void ChannelImportance::Set(std::string name, std::vector<float> values) {
    values_[std::move(name)] = std::move(values);
}

// The importance of `name`'s input channels, or nullptr if it wasn't measured.
const std::vector<float>* ChannelImportance::Find(std::string_view name) const {
    const auto it = values_.find(name);
    return it == values_.end() ? nullptr : &it->second;
}

// Binary file: magic, entry count, then per entry the name (length + bytes)
// and the values (count + floats), little-endian as the machine writes them.
void ChannelImportance::Save(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("importance file: cannot write " + path.string());
    }
    out.write(kMagic, sizeof(kMagic));
    Write(out, static_cast<std::uint32_t>(values_.size()));
    for (const auto& [name, values] : values_) {
        Write(out, static_cast<std::uint32_t>(name.size()));
        out.write(name.data(), static_cast<std::streamsize>(name.size()));
        Write(out, static_cast<std::uint32_t>(values.size()));
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    if (!out) {
        throw std::runtime_error("importance file: write failed");
    }
}

// Reads a file Save wrote. The file steers how INT4 weights are rounded, so
// every size is checked against what is left of the file before anything is
// allocated, and every value must be a finite, non-negative importance.
ChannelImportance ChannelImportance::Load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    char magic[sizeof(kMagic)] = {};
    if (!in.read(magic, sizeof(magic)) || !std::equal(magic, magic + sizeof(magic), kMagic)) {
        throw std::runtime_error("importance file: " + path.string() + " is missing or not an importance file");
    }
    const std::uintmax_t file_size = std::filesystem::file_size(path);
    const auto remaining = [&] { return file_size - static_cast<std::uintmax_t>(in.tellg()); };
    ChannelImportance importance;
    const auto entries = Read<std::uint32_t>(in);
    for (std::uint32_t e = 0; e < entries; ++e) {
        const auto name_size = Read<std::uint32_t>(in);
        if (name_size == 0 || name_size > kMaxNameBytes || name_size > remaining()) {
            throw std::runtime_error("importance file: bad entry name");
        }
        std::string name(name_size, '\0');
        in.read(name.data(), static_cast<std::streamsize>(name.size()));
        const auto count = Read<std::uint32_t>(in);
        if (static_cast<std::uintmax_t>(count) * sizeof(float) > remaining()) {
            throw std::runtime_error("importance file: truncated");
        }
        std::vector<float> values(count);
        in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!std::ranges::all_of(values, [](float v) { return std::isfinite(v) && v >= 0.0f; })) {
            throw std::runtime_error("importance file: " + name + " has a negative or non-finite value");
        }
        importance.Set(std::move(name), std::move(values));
    }
    return importance;
}

// Where `murmur-lab calibrate` saves a model's importance, next to its weights.
std::filesystem::path ImportancePath(const std::filesystem::path& model_dir) {
    return model_dir / "importance.bin";
}

// Adds x_c^2 of `rows` rows of the input `weight` is about to receive (after
// RMSNorm when the options fuse one) to that matrix's sums.
void InputStatistics::Add(const WeightMatrix& weight, const float* x, int rows, const MatVecOptions& options,
                          cudaStream_t stream) {
    Sums& sums = sums_[&weight];
    if (sums.squares.Size() == 0) {
        sums.squares = DeviceBuffer<float>(weight.InFeatures());
        CudaCheck(cudaMemsetAsync(sums.squares.Data(), 0, sums.squares.SizeBytes(), stream));
    }
    AccumulateInputSquares(x, rows, weight.InFeatures(), options.norm_weight, options.norm_epsilon,
                           sums.squares.Data(), stream);
    sums.rows += rows;
}

// Mean x_c^2 per input channel of `weight`; empty if it never saw a row.
// Blocks until the GPU is done.
std::vector<float> InputStatistics::MeanSquares(const WeightMatrix& weight) const {
    const auto it = sums_.find(&weight);
    if (it == sums_.end() || it->second.rows == 0) {
        return {};
    }
    CudaCheck(cudaDeviceSynchronize());
    std::vector<float> mean(it->second.squares.Size());
    it->second.squares.CopyToHost(mean);
    for (float& v : mean) {
        v = static_cast<float>(v / static_cast<double>(it->second.rows));
    }
    return mean;
}

}  // namespace murmur
