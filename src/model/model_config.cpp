#include "model/model_config.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace murmur {

namespace {

using Json = nlohmann::json;

[[noreturn]] void Fail(const std::filesystem::path& path, const std::string& message) {
    throw std::runtime_error(path.string() + ": " + message);
}

// json[key] as a whole number from 1 to the largest int. Anything else
// (zero, negative, fractional, too big, a string) fails naming the key and
// the value found, so a broken config.json says what to fix.
int PositiveInt(const Json& json, const char* key, const std::filesystem::path& path) {
    const Json& value = json.at(key);
    const double number = value.is_number() ? value.get<double>() : 0.0;
    if (!value.is_number_integer() || number < 1.0 || number > std::numeric_limits<int>::max()) {
        Fail(path, std::format("{} must be a positive whole number, not {}", key, value.dump()));
    }
    return value.get<int>();
}

// json[key] as a finite number above 0 that a float holds.
float PositiveFloat(const Json& json, const char* key, const std::filesystem::path& path) {
    const Json& value = json.at(key);
    const double number = value.is_number() ? value.get<double>() : 0.0;
    if (!std::isfinite(number) || number <= 0.0 || number > std::numeric_limits<float>::max()) {
        Fail(path, std::format("{} must be a positive number, not {}", key, value.dump()));
    }
    return static_cast<float>(number);
}

}  // namespace

// Reads config.json and checks that it describes a model this engine can run:
// a Qwen2 decoder with SiLU MLPs, no sliding-window attention, positive
// sizes, head counts that divide evenly, an even head size (rotary
// embeddings turn pairs of numbers) and end-of-text tokens inside the
// vocabulary. Anything else throws, naming the field, rather than loading
// and computing garbage.
ModelConfig ModelConfig::FromFile(const std::filesystem::path& config_json) {
    std::ifstream file(config_json, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open " + config_json.string());
    }
    ModelConfig config;
    try {
        const Json json = Json::parse(file);
        if (json.at("model_type") != "qwen2") {
            Fail(config_json, "unsupported model_type (expected qwen2)");
        }
        if (json.at("hidden_act") != "silu") {
            Fail(config_json, "unsupported hidden_act (expected silu)");
        }
        if (json.value("use_sliding_window", false)) {
            Fail(config_json, "sliding-window attention is not supported");
        }
        config.hidden_size = PositiveInt(json, "hidden_size", config_json);
        config.intermediate_size = PositiveInt(json, "intermediate_size", config_json);
        config.num_layers = PositiveInt(json, "num_hidden_layers", config_json);
        config.num_heads = PositiveInt(json, "num_attention_heads", config_json);
        config.num_kv_heads = PositiveInt(json, "num_key_value_heads", config_json);
        config.vocab_size = PositiveInt(json, "vocab_size", config_json);
        config.max_position_embeddings = PositiveInt(json, "max_position_embeddings", config_json);
        config.rms_norm_eps = PositiveFloat(json, "rms_norm_eps", config_json);
        config.rope_theta = PositiveFloat(json, "rope_theta", config_json);
        config.tie_word_embeddings = json.value("tie_word_embeddings", false);
        const Json& eos = json.at("eos_token_id");
        config.eos_token_ids = eos.is_array() ? eos.get<std::vector<std::int32_t>>()
                                              : std::vector<std::int32_t>{eos.get<std::int32_t>()};
        // Qwen2 derives the head size; a config that states it must agree.
        if (json.contains("head_dim") && json.at("head_dim") != config.hidden_size / config.num_heads) {
            Fail(config_json, std::format("head_dim {} must be hidden_size / num_attention_heads = {} / {}",
                                          json.at("head_dim").dump(), config.hidden_size, config.num_heads));
        }
    } catch (const Json::exception& error) {
        Fail(config_json, std::string("invalid config: ") + error.what());
    }

    if (config.hidden_size % config.num_heads != 0) {
        Fail(config_json, std::format("hidden_size {} must be a multiple of num_attention_heads {}",
                                      config.hidden_size, config.num_heads));
    }
    if (config.num_heads % config.num_kv_heads != 0) {
        Fail(config_json, std::format("num_attention_heads {} must be a multiple of num_key_value_heads {}",
                                      config.num_heads, config.num_kv_heads));
    }
    config.head_dim = config.hidden_size / config.num_heads;
    if (config.head_dim % 2 != 0) {
        Fail(config_json, std::format("the head size, hidden_size / num_attention_heads = {}, must be even for "
                                      "rotary embeddings",
                                      config.head_dim));
    }
    if (config.eos_token_ids.empty()) {
        Fail(config_json, "eos_token_id names no token");
    }
    for (const std::int32_t eos : config.eos_token_ids) {
        if (eos < 0 || eos >= config.vocab_size) {
            Fail(config_json, std::format("eos_token_id {} is outside the vocabulary of {} tokens", eos,
                                          config.vocab_size));
        }
    }
    return config;
}

}  // namespace murmur
