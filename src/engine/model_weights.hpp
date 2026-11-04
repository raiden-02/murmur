#pragma once

#include "common/device_buffer.hpp"
#include "engine/weight_matrix.hpp"
#include "model/model_config.hpp"
#include "model/safetensors.hpp"

#include <vector>

namespace murmur {

class ChannelImportance;
class InputStatistics;

// One decoder layer's weights in GPU memory. Matrices are in the model's
// WeightFormat (the checkpoint's BF16, or quantized to fewer bytes, since
// decode is bound by bytes); norm weights and biases are tiny and kept as
// FP32. Projections that read the same input are stacked into one matrix so
// one kernel computes them.
struct LayerWeights {
    DeviceBuffer<float> attn_norm;   // [hidden]
    WeightMatrix qkv_proj;           // [(heads + 2 * kv_heads) * head_dim, hidden]: q rows, k rows, v rows
    DeviceBuffer<float> qkv_bias;    // [(heads + 2 * kv_heads) * head_dim]
    WeightMatrix o_proj;             // [hidden, heads * head_dim]
    DeviceBuffer<float> mlp_norm;    // [hidden]
    WeightMatrix gate_up_proj;       // [2 * intermediate, hidden]: gate and up rows interleaved (gate 0, up 0, ...)
    WeightMatrix down_proj;          // [hidden, intermediate]
};

struct ModelWeights {
    WeightFormat format = WeightFormat::kBf16;
    WeightMatrix embed_tokens;   // [vocab, hidden]
    std::vector<LayerWeights> layers;
    DeviceBuffer<float> final_norm;   // [hidden]
    WeightMatrix lm_head;             // [vocab, hidden]; empty when tied to embed_tokens

    static ModelWeights Upload(const SafetensorsFile& file, const ModelConfig& config,
                               WeightFormat format = WeightFormat::kBf16,
                               const ChannelImportance* importance = nullptr);
    ChannelImportance CollectImportance(const InputStatistics& statistics) const;

    const WeightMatrix& LmHead() const;
    std::size_t SizeBytes() const;
    std::size_t LargestLayerMatrix() const;
};

}  // namespace murmur
