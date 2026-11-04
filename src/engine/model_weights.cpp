#include "engine/model_weights.hpp"

#include "common/cuda_check.hpp"
#include "engine/channel_importance.hpp"
#include "engine/kernels/elementwise.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace murmur {

namespace {

// Uploads checkpoint tensors, checking each one's shape against the config,
// and converts weight matrices to the model's format on the GPU, steered by
// the importance of each matrix's role when there is one.
class TensorUploader {
public:
    TensorUploader(const SafetensorsFile& file, WeightFormat format, const ChannelImportance* importance)
        : file_(file), format_(format), importance_(importance) {}

    // One or more BF16 weight matrices with the same width, stacked row-wise
    // into one matrix in the order given.
    WeightMatrix StackedMatrix(const std::vector<std::pair<std::string, int>>& parts, int in_features,
                               std::string_view role) {
        int total_rows = 0;
        for (const auto& [name, rows] : parts) {
            total_rows += rows;
        }
        DeviceBuffer<std::uint8_t> bf16(static_cast<std::size_t>(total_rows) * in_features * sizeof(Bf16Bits));
        std::size_t offset = 0;
        for (const auto& [name, rows] : parts) {
            const TensorInfo& tensor = FindBf16(name, {rows, in_features});
            CudaCheck(cudaMemcpy(bf16.Data() + offset, tensor.bytes.data(), tensor.bytes.size(),
                                 cudaMemcpyHostToDevice));
            offset += tensor.bytes.size();
        }
        return WeightMatrix::FromBf16(std::move(bf16), total_rows, in_features, format_, Importance(role, in_features));
    }

    // Two BF16 matrices of the same shape with their rows interleaved (a0,
    // b0, a1, b1, ...), so output pair (2i, 2i + 1) holds row i of each.
    WeightMatrix InterleavedMatrix(const std::string& first, const std::string& second, int rows, int in_features,
                                   std::string_view role) {
        const std::size_t row_bytes = static_cast<std::size_t>(in_features) * sizeof(Bf16Bits);
        DeviceBuffer<std::uint8_t> bf16(2 * rows * row_bytes);
        std::size_t offset = 0;
        for (const std::string& name : {first, second}) {
            const TensorInfo& tensor = FindBf16(name, {rows, in_features});
            CudaCheck(cudaMemcpy2D(bf16.Data() + offset, 2 * row_bytes, tensor.bytes.data(), row_bytes, row_bytes,
                                   rows, cudaMemcpyHostToDevice));
            offset += row_bytes;
        }
        return WeightMatrix::FromBf16(std::move(bf16), 2 * rows, in_features, format_, Importance(role, in_features));
    }

    // One or more vectors (norm weights, biases) concatenated, widened to
    // FP32 if stored as BF16.
    DeviceBuffer<float> StackedVector(const std::vector<std::pair<std::string, int>>& parts) {
        std::size_t total = 0;
        for (const auto& [name, size] : parts) {
            total += static_cast<std::size_t>(size);
        }
        DeviceBuffer<float> out(total);
        std::size_t offset = 0;
        for (const auto& [name, size] : parts) {
            const TensorInfo& tensor = Find(name, {size});
            switch (tensor.dtype) {
            case DType::kF32:
                CudaCheck(cudaMemcpy(out.Data() + offset, tensor.bytes.data(), tensor.bytes.size(),
                                     cudaMemcpyHostToDevice));
                break;
            case DType::kBF16: {
                DeviceBuffer<Bf16Bits> staging(static_cast<std::size_t>(size));
                CudaCheck(cudaMemcpy(staging.Data(), tensor.bytes.data(), tensor.bytes.size(),
                                     cudaMemcpyHostToDevice));
                ConvertBf16ToF32(staging.Data(), out.Data() + offset, staging.Size(), /*stream=*/nullptr);
                CudaCheck(cudaDeviceSynchronize());   // staging is freed at the end of this scope
                break;
            }
            default:
                throw std::runtime_error("weights: " + name + " has unsupported dtype " + DTypeName(tensor.dtype));
            }
            offset += static_cast<std::size_t>(size);
        }
        return out;
    }

    WeightMatrix Matrix(const std::string& name, int out_features, int in_features, std::string_view role) {
        return StackedMatrix({{name, out_features}}, in_features, role);
    }

    DeviceBuffer<float> Vector(const std::string& name, int size) {
        return StackedVector({{name, size}});
    }

private:
    // The measured importance of `role`'s input channels, if any (and if it fits).
    std::span<const float> Importance(std::string_view role, int in_features) const {
        const std::vector<float>* values = importance_ != nullptr ? importance_->Find(role) : nullptr;
        if (values == nullptr || values->size() != static_cast<std::size_t>(in_features)) {
            return {};
        }
        return *values;
    }

    const TensorInfo& Find(const std::string& name, const std::vector<std::int64_t>& shape) {
        const TensorInfo& tensor = file_.Get(name);
        if (tensor.shape != shape) {
            throw std::runtime_error("weights: " + name + " has an unexpected shape");
        }
        return tensor;
    }

    const TensorInfo& FindBf16(const std::string& name, const std::vector<std::int64_t>& shape) {
        const TensorInfo& tensor = Find(name, shape);
        if (tensor.dtype != DType::kBF16) {
            throw std::runtime_error("weights: " + name + " must be BF16, not " + DTypeName(tensor.dtype));
        }
        return tensor;
    }

    const SafetensorsFile& file_;
    WeightFormat format_;
    const ChannelImportance* importance_;
};

std::string LayerRole(int layer, std::string_view matrix) {
    return "layers." + std::to_string(layer) + "." + std::string(matrix);
}

}  // namespace

// Uploads every weight of a Qwen2 checkpoint, converting the matrices
// (embedding table included) to `format`, guided by `importance` when given
// (see ChannelImportance). A mismatched or non-BF16 checkpoint fails here,
// not mid-inference.
ModelWeights ModelWeights::Upload(const SafetensorsFile& file, const ModelConfig& config, WeightFormat format,
                                  const ChannelImportance* importance) {
    const int hidden = config.hidden_size;
    const int q_width = config.QueryWidth();
    const int kv_width = config.KvWidth();
    const int intermediate = config.intermediate_size;

    TensorUploader uploader(file, format, importance);
    ModelWeights weights;
    weights.format = format;
    // Tied, the table is above all the output projection: its importance is the LM head's.
    weights.embed_tokens = uploader.Matrix("model.embed_tokens.weight", config.vocab_size, hidden,
                                           config.tie_word_embeddings ? "lm_head" : "embed");
    weights.layers.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        LayerWeights layer;
        layer.attn_norm = uploader.Vector(p + "input_layernorm.weight", hidden);
        layer.qkv_proj = uploader.StackedMatrix({{p + "self_attn.q_proj.weight", q_width},
                                                 {p + "self_attn.k_proj.weight", kv_width},
                                                 {p + "self_attn.v_proj.weight", kv_width}},
                                                hidden, LayerRole(i, "qkv"));
        layer.qkv_bias = uploader.StackedVector({{p + "self_attn.q_proj.bias", q_width},
                                                 {p + "self_attn.k_proj.bias", kv_width},
                                                 {p + "self_attn.v_proj.bias", kv_width}});
        layer.o_proj = uploader.Matrix(p + "self_attn.o_proj.weight", hidden, q_width, LayerRole(i, "o_proj"));
        layer.mlp_norm = uploader.Vector(p + "post_attention_layernorm.weight", hidden);
        layer.gate_up_proj = uploader.InterleavedMatrix(p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                                                        intermediate, hidden, LayerRole(i, "gate_up"));
        layer.down_proj = uploader.Matrix(p + "mlp.down_proj.weight", hidden, intermediate, LayerRole(i, "down"));
        weights.layers.push_back(std::move(layer));
    }
    weights.final_norm = uploader.Vector("model.norm.weight", hidden);
    if (!config.tie_word_embeddings) {
        weights.lm_head = uploader.Matrix("lm_head.weight", config.vocab_size, hidden, "lm_head");
    }
    return weights;
}

// The output projection: its own matrix, or the embedding table when tied.
const WeightMatrix& ModelWeights::LmHead() const {
    return lm_head.ElementCount() > 0 ? lm_head : embed_tokens;
}

std::size_t ModelWeights::SizeBytes() const {
    std::size_t total = embed_tokens.SizeBytes() + lm_head.SizeBytes() + final_norm.SizeBytes();
    for (const LayerWeights& l : layers) {
        for (const WeightMatrix* m : {&l.qkv_proj, &l.o_proj, &l.gate_up_proj, &l.down_proj}) {
            total += m->SizeBytes();
        }
        for (const DeviceBuffer<float>* v : {&l.attn_norm, &l.qkv_bias, &l.mlp_norm}) {
            total += v->SizeBytes();
        }
    }
    return total;
}

// Each matrix's measured input statistics, by role, for ChannelImportance.
ChannelImportance ModelWeights::CollectImportance(const InputStatistics& statistics) const {
    ChannelImportance importance;
    const auto add = [&](std::string role, const WeightMatrix& m) {
        std::vector<float> values = statistics.MeanSquares(m);
        if (!values.empty()) {
            importance.Set(std::move(role), std::move(values));
        }
    };
    for (int i = 0; i < static_cast<int>(layers.size()); ++i) {
        add(LayerRole(i, "qkv"), layers[i].qkv_proj);
        add(LayerRole(i, "o_proj"), layers[i].o_proj);
        add(LayerRole(i, "gate_up"), layers[i].gate_up_proj);
        add(LayerRole(i, "down"), layers[i].down_proj);
    }
    add("lm_head", LmHead());
    return importance;
}

// Elements in the biggest per-layer matrix: what the exact path's FP32
// scratch must hold.
std::size_t ModelWeights::LargestLayerMatrix() const {
    std::size_t largest = 0;
    for (const LayerWeights& l : layers) {
        for (const WeightMatrix* m : {&l.qkv_proj, &l.o_proj, &l.gate_up_proj, &l.down_proj}) {
            largest = std::max(largest, m->ElementCount());
        }
    }
    return largest;
}

}  // namespace murmur
