#include "engine/model_loader.hpp"

#include "engine/channel_importance.hpp"
#include "model/chat_template.hpp"
#include "model/model_config.hpp"
#include "model/safetensors.hpp"

#include <optional>
#include <stdexcept>

namespace murmur {

// Uploads the weights in model.safetensors to the GPU in `format`; INT4
// uses the model's channel importance (importance.bin, from `murmur-lab
// calibrate`) when it exists. The memory-mapped file is closed again once
// its contents are on the GPU.
ModelWeights LoadWeights(const std::filesystem::path& model_dir, const ModelConfig& config, WeightFormat format) {
    const SafetensorsFile file(model_dir / "model.safetensors");
    std::optional<ChannelImportance> importance;
    if (format == WeightFormat::kInt4 && std::filesystem::exists(ImportancePath(model_dir))) {
        importance = ChannelImportance::Load(ImportancePath(model_dir));
    }
    return ModelWeights::Upload(file, config, format, importance ? &*importance : nullptr);
}

// Reads config.json, tokenizer.json and model.safetensors from `model_dir`
// and builds the transformer with weights in `format`. With a
// memory_budget, the KV cache gets whatever of it the weights leave (in
// place of options.kv_cache_bytes): smaller weights, more cached tokens.
LoadedModel LoadModel(const std::filesystem::path& model_dir, TransformerOptions options, WeightFormat format,
                      std::size_t memory_budget) {
    const ModelConfig config = ModelConfig::FromFile(model_dir / "config.json");
    Tokenizer tokenizer = Tokenizer::FromFile(model_dir / "tokenizer.json");
    if (tokenizer.VocabSize() > config.vocab_size) {
        throw std::runtime_error("tokenizer has more tokens than the model's embedding table");
    }
    ModelWeights weights = LoadWeights(model_dir, config, format);
    if (memory_budget > 0) {
        options.kv_cache_bytes = KvBytesWithin(memory_budget, weights);
    }
    return {std::move(tokenizer), std::make_unique<Transformer>(config, std::move(weights), options)};
}

// The KV cache a memory budget leaves after the weights.
std::size_t KvBytesWithin(std::size_t memory_budget, const ModelWeights& weights) {
    if (weights.SizeBytes() >= memory_budget) {
        throw std::invalid_argument("the memory budget doesn't even hold the weights");
    }
    return memory_budget - weights.SizeBytes();
}

// The draft model for speculative decoding, in `format`, ready to guess for
// up to 8 conversations at once (each with a few hundred cached tokens: a
// small KV cache of its own). It must share the main model's tokenizer.
std::unique_ptr<Speculator> LoadSpeculator(const std::filesystem::path& draft_dir, WeightFormat format,
                                           const Tokenizer& tokenizer) {
    const ModelConfig config = ModelConfig::FromFile(draft_dir / "config.json");
    auto draft = std::make_unique<Transformer>(config, LoadWeights(draft_dir, config, format),
                                               TransformerOptions{.max_batch = 8,
                                                                  .max_logit_rows = 32,
                                                                  .kv_cache_bytes = std::size_t{256} << 20});
    const ChatTemplate chat(tokenizer);
    const TokenId stops[] = {chat.EndOfTurn(), tokenizer.SpecialTokenId("<|endoftext|>")};
    return std::make_unique<Speculator>(std::move(draft), tokenizer.VocabSize(), stops);
}

}  // namespace murmur
