#include "model/model_files.hpp"

#include <cstdlib>

namespace murmur {

namespace {

// The files LoadModel reads from a Hugging Face model folder.
constexpr const char* kModelFiles[] = {"config.json", "tokenizer.json", "model.safetensors"};

// $`variable` if set, otherwise `root`/models/`folder`. An empty root means
// the working directory (the repository root during development).
std::filesystem::path ModelDirectory(const char* variable, const char* folder, const std::filesystem::path& root) {
    char* value = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&value, &length, variable) == 0 && value != nullptr) {
        const std::filesystem::path dir(value);
        std::free(value);
        return dir;
    }
    return root / "models" / folder;
}

}  // namespace

// The main model: $MURMUR_MODEL_DIR or models/qwen2.5-1.5b-instruct.
std::filesystem::path DefaultModelDirectory(const std::filesystem::path& root) {
    return ModelDirectory("MURMUR_MODEL_DIR", "qwen2.5-1.5b-instruct", root);
}

// The speculative-decoding draft: $MURMUR_DRAFT_MODEL_DIR or
// models/qwen2.5-0.5b-instruct (same tokenizer, a third of the layers' width).
std::filesystem::path DraftModelDirectory(const std::filesystem::path& root) {
    return ModelDirectory("MURMUR_DRAFT_MODEL_DIR", "qwen2.5-0.5b-instruct", root);
}

// The files a model folder needs that aren't there, by name; empty when the
// model can be loaded (as far as files go). Lets a program say exactly what
// to download instead of failing on the first missing file.
std::vector<std::string> MissingModelFiles(const std::filesystem::path& model_dir) {
    std::vector<std::string> missing;
    for (const char* file : kModelFiles) {
        std::error_code error;
        if (!std::filesystem::is_regular_file(model_dir / file, error)) {
            missing.emplace_back(file);
        }
    }
    return missing;
}

}  // namespace murmur
