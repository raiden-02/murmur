#include "model/model_config.hpp"

#include "test_files.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace murmur {
namespace {

TEST(ModelConfig, ReadsQwen2Config) {
    const auto path = test::ModelDir() / "config.json";
    MURMUR_SKIP_IF_MISSING(path);
    const ModelConfig config = ModelConfig::FromFile(path);
    EXPECT_EQ(config.hidden_size, 1536);
    EXPECT_EQ(config.intermediate_size, 8960);
    EXPECT_EQ(config.num_layers, 28);
    EXPECT_EQ(config.num_heads, 12);
    EXPECT_EQ(config.num_kv_heads, 2);
    EXPECT_EQ(config.head_dim, 128);
    EXPECT_EQ(config.vocab_size, 151936);
    EXPECT_EQ(config.KvWidth(), 256);
    EXPECT_EQ(config.QueriesPerKvHead(), 6);
    EXPECT_FLOAT_EQ(config.rms_norm_eps, 1e-6f);
    EXPECT_FLOAT_EQ(config.rope_theta, 1e6f);
    EXPECT_TRUE(config.tie_word_embeddings);
}

// A small, valid Qwen2 config.json; each case below breaks one thing in it.
nlohmann::json TinyConfig() {
    return {{"model_type", "qwen2"},     {"hidden_act", "silu"},          {"hidden_size", 64},
            {"intermediate_size", 128},  {"num_hidden_layers", 2},        {"num_attention_heads", 4},
            {"num_key_value_heads", 2},  {"vocab_size", 100},             {"max_position_embeddings", 512},
            {"rms_norm_eps", 1e-6},      {"rope_theta", 10000},           {"eos_token_id", 2}};
}

ModelConfig ReadConfig(const nlohmann::json& json) {
    const auto path = std::filesystem::temp_directory_path() / "murmur_test_config.json";
    std::ofstream(path) << json.dump();
    struct RemoveWhenDone {
        std::filesystem::path path;
        ~RemoveWhenDone() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } cleanup{path};
    return ModelConfig::FromFile(path);
}

TEST(ModelConfig, ReadsAValidConfig) {
    const ModelConfig config = ReadConfig(TinyConfig());
    EXPECT_EQ(config.head_dim, 16);
    EXPECT_EQ(config.QueriesPerKvHead(), 2);
    EXPECT_EQ(config.eos_token_ids, std::vector<std::int32_t>{2});
}

// Every rejection names what is wrong, so a broken config.json says what
// to fix.
TEST(ModelConfig, RejectsWhatItCannotRunAndSaysWhy) {
    struct Case {
        const char* key;
        nlohmann::json value;   // null: the key is removed
        const char* reason;     // part of the error message
    };
    const Case cases[] = {
        {"model_type", "llama", "unsupported model_type"},
        {"hidden_act", "gelu", "unsupported hidden_act"},
        {"use_sliding_window", true, "sliding-window"},
        {"hidden_size", 0, "hidden_size must be a positive whole number, not 0"},
        {"num_hidden_layers", -2, "num_hidden_layers must be a positive whole number, not -2"},
        {"intermediate_size", 1.5, "intermediate_size must be a positive whole number, not 1.5"},
        {"vocab_size", "100", "vocab_size must be a positive whole number"},
        {"max_position_embeddings", 4294967296LL, "max_position_embeddings must be a positive whole number"},
        {"rms_norm_eps", 0, "rms_norm_eps must be a positive number, not 0"},
        {"rope_theta", -1.0, "rope_theta must be a positive number"},
        {"num_attention_heads", 5, "hidden_size 64 must be a multiple of num_attention_heads 5"},
        {"num_key_value_heads", 3, "num_attention_heads 4 must be a multiple of num_key_value_heads 3"},
        {"hidden_size", 36, "must be even for rotary embeddings"},   // 36 / 4 heads = 9
        {"head_dim", 32, "head_dim 32 must be hidden_size / num_attention_heads"},
        {"eos_token_id", 100, "eos_token_id 100 is outside the vocabulary of 100 tokens"},
        {"eos_token_id", nlohmann::json::array(), "eos_token_id names no token"},
        {"rope_theta", nullptr, "invalid config"},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(std::string(c.key) + " = " + c.value.dump());
        nlohmann::json json = TinyConfig();
        if (c.value.is_null()) {
            json.erase(c.key);
        } else {
            json[c.key] = c.value;
        }
        try {
            ReadConfig(json);
            ADD_FAILURE() << "accepted";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find(c.reason), std::string::npos) << error.what();
        }
    }
}

}  // namespace
}  // namespace murmur
