#include "lab/lab.hpp"

#include "common/cuda_check.hpp"
#include "common/device_buffer.hpp"
#include "engine/channel_importance.hpp"
#include "engine/model_loader.hpp"
#include "engine/transformer.hpp"
#include "game/personas.hpp"
#include "lab/logit_comparison.cuh"
#include "model/chat_template.hpp"
#include "model/model_config.hpp"
#include "model/safetensors.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace murmur::lab {

namespace {

constexpr int kWindow = 512;        // tokens per forward pass: one chunk, so every row's logits come back at once
constexpr int kDialogues = 16;      // market NPCs asked one question each
constexpr int kReplyTokens = 96;    // longest reply generated per dialogue
constexpr const char* kProsePath = "reference/eval/prose.txt";
constexpr const char* kQuestions[] = {
    "What's good tonight?", "Have you heard any rumours about the docks?", "How much for that?",
    "Is the market safe after midnight?", "Who should I talk to about a cheap implant?",
};
// Calibration uses other NPCs and other questions than the evaluation, so
// the importance isn't fitted to the text it is judged on.
constexpr int kCalibrationFirstNpc = kDialogues;
constexpr const char* kCalibrationQuestions[] = {
    "Tell me about yourself.", "What happened during the last blackout?", "Can you give me a discount?",
    "Where does the market get its power?", "What do you think of Tessera?",
};

// A candidate: weights in some format, keys and values cached in some
// format, prompt rows multiplied some way.
struct Variant {
    WeightFormat weights;
    KvFormat kv;
    MultiRowMath math;
};

// First each cost on its own (everything else exact), then the combinations
// the engine serves: prompt rows on the fastest tensor-core path for the
// format (MultiRowMath::kFastest), a compact cache.
constexpr Variant kVariants[] = {
    {WeightFormat::kBf16, KvFormat::kFp32, MultiRowMath::kCublasBf16},
    {WeightFormat::kBf16, KvFormat::kFp32, MultiRowMath::kTensorCores},
    {WeightFormat::kFp8, KvFormat::kFp32, MultiRowMath::kExactFp32},
    {WeightFormat::kFp8, KvFormat::kFp32, MultiRowMath::kFp8TensorCores},
    {WeightFormat::kInt4, KvFormat::kFp32, MultiRowMath::kExactFp32},
    {WeightFormat::kBf16, KvFormat::kBf16, MultiRowMath::kExactFp32},
    {WeightFormat::kBf16, KvFormat::kFp8, MultiRowMath::kExactFp32},
    {WeightFormat::kBf16, KvFormat::kBf16, MultiRowMath::kFastest},
    {WeightFormat::kFp8, KvFormat::kBf16, MultiRowMath::kFastest},
    {WeightFormat::kInt4, KvFormat::kBf16, MultiRowMath::kFastest},
    {WeightFormat::kFp8, KvFormat::kFp8, MultiRowMath::kFastest},
    {WeightFormat::kInt4, KvFormat::kFp8, MultiRowMath::kFastest},
};

// The model's private cache, switched to `format` if it isn't already.
void UseKvFormat(Transformer& model, KvFormat format) {
    if (model.Options().kv_format != format) {
        model.ResizeKvCache(model.Options().kv_cache_bytes, format);
    }
}

// Copies the logits checkpoint of a forward pass (every row) into `target`.
class LogitsCapture : public ForwardObserver {
public:
    explicit LogitsCapture(float* target) : target_(target) {}

    void OnCheckpoint(std::string_view name, const float* device_data, int rows, int cols, int row_stride) override {
        if (name == "logits") {
            CudaCheck(cudaMemcpy2D(target_, cols * sizeof(float), device_data, row_stride * sizeof(float),
                                   cols * sizeof(float), rows, cudaMemcpyDeviceToDevice));
        }
    }

private:
    float* target_;
};

// One scored window of tokens. A dialogue's is its prompt followed by the
// reference's reply, cut to kWindow tokens (which may cut the reply short).
struct Sequence {
    std::vector<TokenId> tokens;
    bool dialogue = false;
    std::size_t prompt_size = 0;   // dialogues: tokens[0, prompt_size) is the prompt, whole
    std::vector<TokenId> reply;    // dialogues: the reference's whole greedy reply
};

struct Totals {
    std::vector<float> kl;
    double nll[2] = {0.0, 0.0};   // [prose, dialogue], candidate
    double reference_nll[2] = {0.0, 0.0};
    int scored[2] = {0, 0};
    int same_top = 0;
    int identical_replies = 0;
    double prefix_tokens = 0.0;
};

double Percentile(std::vector<float> values, double fraction) {
    std::ranges::sort(values);
    return values.empty() ? 0.0 : values[static_cast<std::size_t>(fraction * (values.size() - 1))];
}

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("quality: cannot read " + path.string());
    }
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

TransformerOptions EvaluationOptions() {
    return {.max_context = 1024,
            .max_chunk = kWindow,
            .max_batch = 1,
            .kv_cache_bytes = std::size_t{64} << 20,
            .kv_format = KvFormat::kFp32,
            .multi_row_math = MultiRowMath::kExactFp32};
}

// Greedy reply to `prompt`, generated one token at a time (the decode path).
std::vector<TokenId> GreedyReply(Transformer& model, std::span<const TokenId> prompt, int vocab, TokenId end) {
    model.Forward(prompt, 0);
    std::vector<TokenId> reply;
    int position = static_cast<int>(prompt.size());
    while (static_cast<int>(reply.size()) < kReplyTokens) {
        const TokenId next = model.ArgmaxLastLogits(vocab);
        reply.push_back(next);
        if (next == end) {
            break;
        }
        model.Forward(std::span(&next, 1), position++);
    }
    return reply;
}

}  // namespace

// How much each weight format and multi-row math costs in quality, against
// the original model (BF16 weights, exact FP32 math). Every position of two
// text sets runs through the reference and through each candidate:
//   prose:    a fixed excerpt of this repository's docs (reference/eval/prose.txt),
//   dialogue: 16 market NPCs answering a question, with the reference's own
//             greedy replies.
// Reports, per candidate: the mean and p99 KL divergence of its next-token
// distribution from the reference's, how often both rank the same token
// first, perplexity on each set (e^mean negative log-likelihood of the
// actual next token), and, generating replies token by token as the game
// does, how many replies come out identical and how long the shared start is.
//
// INT4 uses the model's importance file (see `calibrate`) when there is one.
int RunQuality(Args args) {
    std::string json_path;
    bool no_importance = false;
    ArgParser parser("murmur-lab quality");
    parser.Text("--json", "<path>", json_path, "also write the results there")
        .Switch("--no-importance", no_importance, "quantize INT4 without the model's importance file");
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }
    const bool use_importance = !no_importance;

    const std::filesystem::path dir = DefaultModelDirectory();
    const ModelConfig config = ModelConfig::FromFile(dir / "config.json");
    const Tokenizer tokenizer = Tokenizer::FromFile(dir / "tokenizer.json");
    const ChatTemplate chat(tokenizer);
    const int vocab = tokenizer.VocabSize();
    std::optional<ChannelImportance> importance;
    if (use_importance && std::filesystem::exists(ImportancePath(dir))) {
        importance = ChannelImportance::Load(ImportancePath(dir));
    }
    std::println("INT4 importance: {}", importance ? ImportancePath(dir).string() : "none (run `calibrate`)");
    std::map<WeightFormat, std::unique_ptr<Transformer>> models;
    {
        const SafetensorsFile file(dir / "model.safetensors");
        for (const WeightFormat format : {WeightFormat::kBf16, WeightFormat::kFp8, WeightFormat::kInt4}) {
            models[format] = std::make_unique<Transformer>(
                config, ModelWeights::Upload(file, config, format, importance ? &*importance : nullptr),
                EvaluationOptions());
        }
    }
    Transformer& reference = *models.at(WeightFormat::kBf16);

    std::vector<Sequence> sequences;
    const std::vector<TokenId> prose = tokenizer.Encode(ReadText(kProsePath), SpecialTokens::kAsText);
    for (std::size_t begin = 0; begin + 2 <= prose.size(); begin += kWindow) {
        const std::size_t end = std::min(prose.size(), begin + kWindow);
        sequences.push_back({.tokens = {prose.begin() + begin, prose.begin() + end}});
    }
    const int prose_windows = static_cast<int>(sequences.size());
    for (int i = 0; i < kDialogues; ++i) {
        const ChatMessage messages[] = {
            {ChatRole::kSystem, personas::MarketSystemPrompt(personas::kMarketCast[i % personas::kMarketCast.size()])},
            {ChatRole::kUser, kQuestions[i % std::size(kQuestions)]}};
        Sequence s{.tokens = chat.EncodeConversation(messages, true), .dialogue = true};
        s.prompt_size = s.tokens.size();
        if (s.prompt_size >= static_cast<std::size_t>(kWindow)) {
            throw std::runtime_error("quality: a dialogue's prompt fills the whole window");
        }
        s.reply = GreedyReply(reference, s.tokens, vocab, chat.EndOfTurn());
        s.tokens.insert(s.tokens.end(), s.reply.begin(), s.reply.end());
        s.tokens.resize(std::min<std::size_t>(s.tokens.size(), kWindow));
        sequences.push_back(std::move(s));
    }
    std::println("{} prose windows ({} tokens), {} dialogues; {} candidates\n", prose_windows, prose.size(),
                 kDialogues, std::size(kVariants));

    const std::size_t logits_floats = static_cast<std::size_t>(kWindow) * config.vocab_size;
    DeviceBuffer<float> reference_logits(logits_floats);
    DeviceBuffer<float> candidate_logits(logits_floats);
    DeviceBuffer<int> next_tokens(kWindow);
    DeviceBuffer<PositionQuality> positions(kWindow);
    std::vector<Totals> totals(std::size(kVariants));

    for (const Sequence& s : sequences) {
        const int n = static_cast<int>(s.tokens.size());
        std::vector<int> next(kWindow, -1);
        std::copy(s.tokens.begin() + 1, s.tokens.end(), next.begin());
        next_tokens.CopyFromHost(next);
        LogitsCapture reference_capture(reference_logits.Data());
        UseKvFormat(reference, KvFormat::kFp32);
        reference.SetMultiRowMath(MultiRowMath::kExactFp32);
        reference.Forward(s.tokens, 0, LogitsRows::kAll, &reference_capture);

        for (std::size_t v = 0; v < std::size(kVariants); ++v) {
            Transformer& model = *models.at(kVariants[v].weights);
            UseKvFormat(model, kVariants[v].kv);
            model.SetMultiRowMath(kVariants[v].math);
            LogitsCapture capture(candidate_logits.Data());
            model.Forward(s.tokens, 0, LogitsRows::kAll, &capture);
            CompareLogits(reference_logits.Data(), candidate_logits.Data(), n, config.vocab_size, vocab,
                          next_tokens.Data(), positions.Data(), nullptr);
            std::vector<PositionQuality> result(kWindow);
            positions.CopyToHost(result);
            Totals& t = totals[v];
            const int set = s.dialogue ? 1 : 0;
            for (int p = 0; p < n - 1; ++p) {   // the last position has no next token to score
                t.kl.push_back(result[p].kl);
                t.same_top += result[p].same_top_token;
                t.nll[set] += result[p].nll_candidate;
                t.reference_nll[set] += result[p].nll_reference;
                ++t.scored[set];
            }
        }
    }

    // Replies generated token by token, as served: prompts on tensor cores.
    for (std::size_t v = 0; v < std::size(kVariants); ++v) {
        if (kVariants[v].math != MultiRowMath::kFastest) {
            continue;   // served combinations only
        }
        Transformer& model = *models.at(kVariants[v].weights);
        UseKvFormat(model, kVariants[v].kv);
        model.SetMultiRowMath(kVariants[v].math);
        for (const Sequence& s : sequences) {
            if (!s.dialogue) {
                continue;
            }
            const std::span<const TokenId> prompt(s.tokens.data(), s.prompt_size);
            const std::vector<TokenId> reply = GreedyReply(model, prompt, vocab, chat.EndOfTurn());
            const auto [a, b] = std::ranges::mismatch(reply, s.reply);
            totals[v].prefix_tokens += static_cast<double>(a - reply.begin()) / kDialogues;
            totals[v].identical_replies += reply == s.reply ? 1 : 0;
        }
    }

    const double reference_prose = std::exp(totals[0].reference_nll[0] / totals[0].scored[0]);
    const double reference_dialogue = std::exp(totals[0].reference_nll[1] / totals[0].scored[1]);
    std::println("reference: bf16 weights, fp32 KV cache, exact FP32 math, {:.2f} GB; perplexity prose {:.3f}, "
                 "dialogue {:.3f}\n",
                 reference.WeightBytes() / 1e9, reference_prose, reference_dialogue);
    std::println("{:<7} {:<5} {:<6} {:>6} {:>9} {:>8} {:>7} {:>9} {:>10} {:>10} {:>7}", "weights", "kv", "math", "GB",
                 "KL mean", "KL p99", "top-1", "ppl prose", "ppl dialog", "same reply", "shared");
    nlohmann::json json = {{"reference_perplexity", {{"prose", reference_prose}, {"dialogue", reference_dialogue}}},
                           {"reference_gb", reference.WeightBytes() / 1e9},
                           {"positions", totals[0].kl.size()},
                           {"candidates", nlohmann::json::array()}};
    for (std::size_t v = 0; v < std::size(kVariants); ++v) {
        const Totals& t = totals[v];
        double kl_sum = 0.0;
        for (const float kl : t.kl) {
            kl_sum += kl;
        }
        const double kl_mean = kl_sum / t.kl.size();
        const double kl_p99 = Percentile(t.kl, 0.99);
        const double top1 = 100.0 * t.same_top / t.kl.size();
        const double ppl_prose = std::exp(t.nll[0] / t.scored[0]);
        const double ppl_dialogue = std::exp(t.nll[1] / t.scored[1]);
        const Variant& variant = kVariants[v];
        const double gb = models.at(variant.weights)->WeightBytes() / 1e9;
        const bool generated = variant.math == MultiRowMath::kFastest;
        std::println("{:<7} {:<5} {:<6} {:>6.2f} {:>9.5f} {:>8.4f} {:>6.1f}% {:>9.3f} {:>10.3f} {:>10} {:>7}",
                     WeightFormatName(variant.weights), KvFormatName(variant.kv), MultiRowMathName(variant.math), gb,
                     kl_mean, kl_p99, top1, ppl_prose, ppl_dialogue,
                     generated ? std::format("{}/{}", t.identical_replies, kDialogues) : "",
                     generated ? std::format("{:.0f}", t.prefix_tokens) : "");
        json["candidates"].push_back({{"weights", WeightFormatName(variant.weights)},
                                      {"kv", KvFormatName(variant.kv)},
                                      {"math", MultiRowMathName(variant.math)},
                                      {"gb", gb},
                                      {"kl_mean", kl_mean},
                                      {"kl_p99", kl_p99},
                                      {"top1_percent", top1},
                                      {"perplexity_prose", ppl_prose},
                                      {"perplexity_dialogue", ppl_dialogue},
                                      {"identical_replies", generated ? t.identical_replies : -1},
                                      {"shared_prefix_tokens", generated ? t.prefix_tokens : -1.0}});
    }
    if (!json_path.empty()) {
        std::ofstream(json_path) << json.dump(1) << '\n';
        std::println("\nwrote {}", json_path);
    }
    return EXIT_SUCCESS;
}

// Measures the model's ChannelImportance for INT4: runs the original model
// over market dialogues (NPCs and questions that `quality` doesn't use),
// prompts and greedy replies, recording how strongly every input channel of
// every matrix is driven, and saves it next to the weights, where loading
// INT4 weights picks it up.
int RunCalibrate(Args args) {
    if (const std::optional<int> exit_code = ArgParser("murmur-lab calibrate").Parse(args)) {
        return *exit_code;
    }
    const std::filesystem::path dir = DefaultModelDirectory();
    LoadedModel model = LoadModel(dir, EvaluationOptions());
    Transformer& transformer = *model.transformer;
    const ChatTemplate chat(model.tokenizer);
    InputStatistics statistics;
    transformer.SetInputStatistics(&statistics);
    int tokens = 0;
    const int npcs = static_cast<int>(personas::kMarketCast.size()) - kCalibrationFirstNpc;
    for (int i = 0; i < npcs; ++i) {
        for (const char* question : kCalibrationQuestions) {
            const ChatMessage messages[] = {
                {ChatRole::kSystem, personas::MarketSystemPrompt(personas::kMarketCast[kCalibrationFirstNpc + i])},
                {ChatRole::kUser, question}};
            const std::vector<TokenId> prompt = chat.EncodeConversation(messages, true);
            tokens += static_cast<int>(prompt.size() + GreedyReply(transformer, prompt, model.tokenizer.VocabSize(),
                                                                   chat.EndOfTurn())
                                                           .size());
        }
    }
    transformer.SetInputStatistics(nullptr);
    const ChannelImportance importance = transformer.Weights().CollectImportance(statistics);
    importance.Save(ImportancePath(dir));
    std::println("{} dialogues, {} tokens: importance of every matrix written to {}", npcs * std::size(kCalibrationQuestions),
                 tokens, ImportancePath(dir).string());
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
