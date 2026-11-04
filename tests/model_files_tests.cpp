#include "model/model_files.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace murmur {
namespace {

TEST(ModelFiles, NamesEveryFileAFolderLacks) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "murmur_test_model_files";
    std::filesystem::remove_all(dir);
    EXPECT_EQ(MissingModelFiles(dir), (std::vector<std::string>{"config.json", "tokenizer.json", "model.safetensors"}));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "config.json") << "{}";
    std::filesystem::create_directory(dir / "model.safetensors");   // a folder of that name isn't the file
    EXPECT_EQ(MissingModelFiles(dir), (std::vector<std::string>{"tokenizer.json", "model.safetensors"}));
    std::filesystem::remove_all(dir);
}

// $MURMUR_MODEL_DIR wins over the default folder, which is relative to the
// given root.
TEST(ModelFiles, TheEnvironmentPointsAtAnotherCopy) {
    char* saved = nullptr;
    std::size_t length = 0;
    ASSERT_EQ(_dupenv_s(&saved, &length, "MURMUR_MODEL_DIR"), 0);
    const std::string previous = saved != nullptr ? saved : "";
    std::free(saved);

    ASSERT_EQ(_putenv_s("MURMUR_MODEL_DIR", ""), 0);   // an empty value unsets it on Windows
    EXPECT_EQ(DefaultModelDirectory("root"), std::filesystem::path("root") / "models" / "qwen2.5-1.5b-instruct");
    ASSERT_EQ(_putenv_s("MURMUR_MODEL_DIR", "D:/elsewhere/qwen"), 0);
    EXPECT_EQ(DefaultModelDirectory("root"), std::filesystem::path("D:/elsewhere/qwen"));
    ASSERT_EQ(_putenv_s("MURMUR_MODEL_DIR", previous.c_str()), 0);
}

}  // namespace
}  // namespace murmur
