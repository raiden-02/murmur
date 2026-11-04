#pragma once

#include "model/unicode.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <string_view>

// A tokenizer.json small enough to build in a test: the 256 single bytes,
// merges that make a few action words single tokens (" walk" included, as in
// Qwen's vocabulary) and the chat template's special tokens. Tests of pure
// logic over tokens (token grammars) run on it without the downloaded model.
namespace murmur::test {

// How byte-level BPE spells byte `b` as a character (see tokenizer.cpp):
// printable bytes stand for themselves, the rest take U+0100 onwards in order.
inline char32_t ByteLevelCharacter(int b) {
    char32_t next_substitute = 256;
    for (int other = 0; other < 256; ++other) {
        const bool printable = (other >= '!' && other <= '~') || (other >= 0xA1 && other <= 0xAC) ||
                               (other >= 0xAE && other <= 0xFF);
        const char32_t character = printable ? static_cast<char32_t>(other) : next_substitute++;
        if (other == b) {
            return character;
        }
    }
    return 0;
}

inline std::string ByteLevelSpelling(std::string_view bytes) {
    std::string spelling;
    for (const char c : bytes) {
        unicode::AppendUtf8(ByteLevelCharacter(static_cast<unsigned char>(c)), spelling);
    }
    return spelling;
}

// Writes the tiny tokenizer to a temporary file and returns its path.
inline std::filesystem::path WriteTinyTokenizer() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int b = 0; b < 256; ++b) {
        vocab[ByteLevelSpelling(std::string(1, static_cast<char>(b)))] = b;
    }
    int next_id = 256;
    nlohmann::json merges = nlohmann::json::array();
    // Each word is built left to right, one merge per byte. Words with a
    // leading space come first, so their merges rank before the bare word's
    // and " walk" ends as one token rather than " " + "walk".
    for (const std::string_view word : {" walk", " none", "walk", "none", "sell", "hack", "noodles", "DOCKS"}) {
        std::string built(word.substr(0, 1));
        for (const char c : word.substr(1)) {
            const std::string merged = built + c;
            if (!vocab.contains(ByteLevelSpelling(merged))) {
                vocab[ByteLevelSpelling(merged)] = next_id++;
                merges.push_back(ByteLevelSpelling(built) + " " + ByteLevelSpelling(std::string(1, c)));
            }
            built = merged;
        }
    }
    nlohmann::json added = nlohmann::json::array();
    for (const char* special : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) {
        added.push_back({{"id", next_id++}, {"content", special}, {"special", true}});
    }
    const nlohmann::json tokenizer = {
        {"added_tokens", added},
        {"normalizer", {{"type", "NFC"}}},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           {{{"type", "Split"},
             {"pattern",
              {{"Regex", R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", false}, {"use_regex", false}}}}}},
        {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", merges}}},
    };
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "murmur_tiny_tokenizer.json";
    std::ofstream(path, std::ios::binary) << tokenizer.dump();
    return path;
}

}  // namespace murmur::test
