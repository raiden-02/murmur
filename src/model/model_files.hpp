#pragma once

// Where the downloaded models live, and whether a folder holds all of one.

#include <filesystem>
#include <string>
#include <vector>

namespace murmur {

std::filesystem::path DefaultModelDirectory(const std::filesystem::path& root = {});
std::filesystem::path DraftModelDirectory(const std::filesystem::path& root = {});
std::vector<std::string> MissingModelFiles(const std::filesystem::path& model_dir);

}  // namespace murmur
