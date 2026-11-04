#pragma once

#include "engine/speculator.hpp"
#include "engine/transformer.hpp"
#include "model/model_files.hpp"   // where the model folders are, for every caller of LoadModel
#include "model/tokenizer.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace murmur {

// Everything needed to run a model, loaded from a Hugging Face model folder.
struct LoadedModel {
    Tokenizer tokenizer;
    std::unique_ptr<Transformer> transformer;
};

LoadedModel LoadModel(const std::filesystem::path& model_dir, TransformerOptions options = {},
                      WeightFormat format = WeightFormat::kBf16, std::size_t memory_budget = 0);

ModelWeights LoadWeights(const std::filesystem::path& model_dir, const ModelConfig& config, WeightFormat format);

std::size_t KvBytesWithin(std::size_t memory_budget, const ModelWeights& weights);

std::unique_ptr<Speculator> LoadSpeculator(const std::filesystem::path& draft_dir, WeightFormat format,
                                           const Tokenizer& tokenizer);

}  // namespace murmur
