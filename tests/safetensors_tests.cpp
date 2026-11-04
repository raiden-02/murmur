#include "model/safetensors.hpp"

#include "test_files.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace murmur {
namespace {

// Writes a safetensors file from a JSON header and raw data, deleting it
// when the test ends.
class TempSafetensors {
public:
    TempSafetensors(const std::string& header, const std::vector<std::uint8_t>& data,
                    std::uint64_t header_size_override = 0)
        : path_(std::filesystem::temp_directory_path() /
                ("murmur_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".safetensors")) {
        std::ofstream file(path_, std::ios::binary);
        const std::uint64_t size = header_size_override != 0 ? header_size_override : header.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file << header;
        file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    ~TempSafetensors() { std::filesystem::remove(path_); }
    TempSafetensors(const TempSafetensors&) = delete;
    TempSafetensors& operator=(const TempSafetensors&) = delete;

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::vector<std::uint8_t> FloatBytes(const std::vector<float>& values) {
    std::vector<std::uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

TEST(Safetensors, ReadsTensorsAndMetadata) {
    const TempSafetensors file(
        R"({"__metadata__":{"format":"pt"},)"
        R"("b":{"dtype":"F32","shape":[2],"data_offsets":[12,20]},)"
        R"("a":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}})",
        FloatBytes({1, 2, 3, 4, 5}));
    const SafetensorsFile st(file.path());

    ASSERT_EQ(st.Tensors().size(), 2u);
    EXPECT_EQ(st.Tensors()[0].name, "a");   // sorted by name
    const TensorInfo& b = st.Get("b");
    EXPECT_EQ(b.dtype, DType::kF32);
    EXPECT_EQ(b.shape, (std::vector<std::int64_t>{2}));
    EXPECT_EQ(b.ElementCount(), 2);
    float values[2];
    std::memcpy(values, b.bytes.data(), sizeof(values));
    EXPECT_EQ(values[0], 4.0f);
    EXPECT_EQ(values[1], 5.0f);
    EXPECT_EQ(st.Find("missing"), nullptr);
    EXPECT_THROW(static_cast<void>(st.Get("missing")), std::out_of_range);
}

TEST(Safetensors, RejectsMalformedFiles) {
    const auto data = FloatBytes({1, 2});
    const auto expect_rejected = [&](const std::string& header, std::uint64_t size_override = 0) {
        const TempSafetensors file(header, data, size_override);
        EXPECT_THROW(SafetensorsFile{file.path()}, std::runtime_error) << header;
    };
    const std::string ok = R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";

    expect_rejected(ok, 1ull << 40);                                                       // header size past EOF
    expect_rejected(R"({"a":)");                                                           // truncated JSON
    expect_rejected(R"([1,2])");                                                           // not an object
    expect_rejected(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,12]}})");         // past the data
    expect_rejected(R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})");          // shape != bytes
    expect_rejected(R"({"a":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}})");         // negative dim
    expect_rejected(R"({"a":{"dtype":"Q4","shape":[2],"data_offsets":[0,8]}})");           // unknown dtype
    expect_rejected(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}})");          // reversed range
    expect_rejected(R"({"a":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,8]}})");  // overflow
    expect_rejected(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
                    R"("b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})");          // overlap
}

TEST(Safetensors, OpensTheModelWeights) {
    const auto path = test::ModelDir() / "model.safetensors";
    MURMUR_SKIP_IF_MISSING(path);
    const SafetensorsFile st(path);
    const TensorInfo& embed = st.Get("model.embed_tokens.weight");
    EXPECT_EQ(embed.dtype, DType::kBF16);
    EXPECT_EQ(embed.shape, (std::vector<std::int64_t>{151936, 1536}));
    EXPECT_NE(st.Find("model.layers.27.mlp.down_proj.weight"), nullptr);
}

}  // namespace
}  // namespace murmur
