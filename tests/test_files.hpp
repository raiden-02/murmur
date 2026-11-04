#pragma once

#include "model/model_files.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

// Locations of the downloaded models and the reference dumps written by
// reference/dump_reference.py. Tests that need them skip when they're absent.
namespace murmur::test {

inline std::filesystem::path SourcePath(const std::string& relative) {
    return std::filesystem::path(MURMUR_SOURCE_DIR) / relative;
}

// $MURMUR_MODEL_DIR, as for the game and the lab, else the repository's
// models/ folder (wherever the tests run from).
inline std::filesystem::path ModelDir() {
    return DefaultModelDirectory(MURMUR_SOURCE_DIR);
}

// The small model that drafts tokens for speculative decoding:
// $MURMUR_DRAFT_MODEL_DIR, else the repository's models/ folder.
inline std::filesystem::path DraftModelDir() {
    return DraftModelDirectory(MURMUR_SOURCE_DIR);
}

inline std::filesystem::path ReferenceDir() {
    return SourcePath("reference/out");
}

inline nlohmann::json ReadJson(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open " + path.string());
    }
    return nlohmann::json::parse(file);
}

// Whether to assert on wall-clock time. Off by default: a busy machine (a
// build, a game, the desktop sharing the GPU) can slow any run down, and a
// test that fails for that says nothing about the code. Set
// MURMUR_TIMING_TESTS=1 on a quiet machine to check the time budgets too.
inline bool TimingChecksEnabled() {
    char* value = nullptr;
    std::size_t length = 0;
    const bool set = _dupenv_s(&value, &length, "MURMUR_TIMING_TESTS") == 0 && value != nullptr &&
                     std::string(value) == "1";
    std::free(value);
    return set;
}

}  // namespace murmur::test

// Skips the current test when `path` doesn't exist (e.g. the model weights
// haven't been downloaded on this machine).
#define MURMUR_SKIP_IF_MISSING(path)                                                 \
    do {                                                                             \
        if (!std::filesystem::exists(path)) {                                        \
            GTEST_SKIP() << "missing " << (path).string();                           \
        }                                                                            \
    } while (false)
