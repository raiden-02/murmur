#include "engine/transformer.hpp"
#include "model/model_config.hpp"
#include "model/safetensors.hpp"

#include "reference_comparer.hpp"
#include "test_files.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace murmur {
namespace {

// The 0.5B draft model (a different shape: 24 layers, hidden 896, 14 query
// heads of 64) runs on the same engine and must match its own Hugging Face
// reference the way the main model does.
class DraftModelTest : public ::testing::TestWithParam<const char*> {
protected:
    static void SetUpTestSuite() {
        const auto dir = test::DraftModelDir();
        if (!std::filesystem::exists(dir / "model.safetensors") ||
            !std::filesystem::exists(test::ReferenceDir() / "draft_plain.json")) {
            return;
        }
        const ModelConfig config = ModelConfig::FromFile(dir / "config.json");
        const SafetensorsFile weights_file(dir / "model.safetensors");
        model_ = std::make_unique<Transformer>(config, ModelWeights::Upload(weights_file, config),
                                               TransformerOptions{.kv_format = KvFormat::kFp32,
                                                                  .multi_row_math = MultiRowMath::kExactFp32});
    }
    static void TearDownTestSuite() { model_.reset(); }
    void SetUp() override {
        if (!model_) {
            GTEST_SKIP() << "draft model or its reference dumps not present";
        }
    }

    static std::vector<TokenId> Ids(const std::string& name, const char* key) {
        return test::ReadJson(test::ReferenceDir() / (name + ".json")).at(key).get<std::vector<TokenId>>();
    }

    static inline std::unique_ptr<Transformer> model_;
};

TEST_P(DraftModelTest, EveryCheckpointMatchesTheReference) {
    const std::string name = GetParam();
    const SafetensorsFile reference(test::ReferenceDir() / (name + ".safetensors"));
    test::ReferenceComparer comparer(reference);
    model_->Forward(Ids(name, "input_ids"), 0, LogitsRows::kAll, &comparer);
    ASSERT_EQ(comparer.Errors().size(), 2 + 6 * model_->Config().num_layers);
    for (const auto& error : comparer.Errors()) {
        EXPECT_LT(error.value, 1e-3) << error.name;
    }
}

TEST_P(DraftModelTest, GreedyDecodingReproducesTheReference) {
    const std::string name = GetParam();
    const std::vector<TokenId> prompt = Ids(name, "input_ids");
    const std::vector<TokenId> expected = Ids(name, "greedy_ids");
    model_->Forward(prompt, 0);
    int position = static_cast<int>(prompt.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const TokenId next = model_->ArgmaxLastLogits(151665);
        ASSERT_EQ(next, expected[i]) << "first difference at generated token " << i;
        model_->Forward(std::span(&next, 1), position++);
    }
}

INSTANTIATE_TEST_SUITE_P(Prompts, DraftModelTest, ::testing::Values("draft_plain", "draft_noodle_vendor"));

}  // namespace
}  // namespace murmur
