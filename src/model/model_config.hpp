#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace murmur {

// The shape of a Qwen2-family decoder, read from a Hugging Face config.json.
struct ModelConfig {
    int hidden_size = 0;          // width of each token's vector
    int intermediate_size = 0;    // MLP width
    int num_layers = 0;
    int num_heads = 0;            // query heads
    int num_kv_heads = 0;         // key/value heads (grouped-query attention)
    int head_dim = 0;
    int vocab_size = 0;           // rows in the embedding table
    int max_position_embeddings = 0;
    float rms_norm_eps = 0.0f;
    float rope_theta = 0.0f;
    bool tie_word_embeddings = false;
    std::vector<std::int32_t> eos_token_ids;

    static ModelConfig FromFile(const std::filesystem::path& config_json);

    int QueryWidth() const { return num_heads * head_dim; }
    int KvWidth() const { return num_kv_heads * head_dim; }
    int QueriesPerKvHead() const { return num_heads / num_kv_heads; }
};

}  // namespace murmur
