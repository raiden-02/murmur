#pragma once

#include "common/cuda_graph.hpp"
#include "common/cuda_stream.hpp"
#include "common/device_buffer.hpp"
#include "common/pinned_buffer.hpp"
#include "engine/kernels/decode_attention.cuh"
#include "engine/kv_cache.hpp"
#include "engine/kv_cache_manager.hpp"
#include "engine/linear.hpp"
#include "engine/model_weights.hpp"
#include "model/model_config.hpp"
#include "model/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace murmur {

// Receives intermediate activations during a forward pass, by the same names
// as the reference dumps ("embed", "layers.3.q", ..., "logits"). Fused
// kernels never materialize some of the reference's tensors (the norms and
// the projections before their residual add), so those aren't reported.
// For testing and inspection; costs a GPU synchronization per checkpoint.
class ForwardObserver {
public:
    virtual ~ForwardObserver() = default;
    // `device_data` is rows x cols floats in GPU memory, consecutive rows
    // `row_stride` floats apart, valid during the call.
    virtual void OnCheckpoint(std::string_view name, const float* device_data, int rows, int cols,
                              int row_stride) = 0;
};

// Which rows of logits a single-sequence forward pass computes.
enum class LogitsRows {
    kLast,   // only the last token's: all that generation needs
    kAll,    // every token's: for comparing with the reference
};

struct TransformerOptions {
    int max_context = 4096;                         // longest sequence
    int max_chunk = 512;                            // tokens per forward pass, all sequences together
    int max_batch = 64;                             // sequences per forward pass
    int max_logit_rows = 80;                        // logits rows per pass: one per sequence, more to check drafts
    int page_size = 16;                             // tokens per KV-cache page (a power of two)
    std::size_t kv_cache_bytes = std::size_t{1} << 30;   // GPU memory for the paged KV cache
    KvFormat kv_format = KvFormat::kBf16;                // how cached keys and values are stored
    bool use_cuda_graphs = true;                    // replay decode steps as recorded graphs
    MultiRowMath multi_row_math = MultiRowMath::kFastest;           // several rows per matrix product
};

// One sequence's part of a batched forward pass: its next tokens, at
// positions sequence->length, ... The sequence's pages must already cover
// them (KvCacheManager::Reserve); advancing its length afterwards is the
// caller's job.
struct StepEntry {
    const KvSequence* sequence = nullptr;
    std::span<const TokenId> tokens;
    int logits = 1;   // next-token logits after each of the last `logits` tokens: 1 to generate, 0 for a
                      // prompt chunk that isn't the last, more to check speculative drafts
    bool decode_each_token = false;   // run every token as its own decode row, attending to the cache up to
                                      // itself (checking guesses: the same attention as decoding them one by
                                      // one, and steps of decode rows only replay as CUDA graphs)
};

// The model's forward pass on the GPU: weights, the paged KV cache,
// preallocated activations and the stream all the work runs on. Step runs
// many sequences at once (decode tokens and prompt chunks mixed); Forward is
// the single-sequence convenience on a private sequence.
class Transformer {
public:
    Transformer(const ModelConfig& config, ModelWeights weights, TransformerOptions options = {});

    void Step(std::span<const StepEntry> entries, ForwardObserver* observer = nullptr);
    float* StepLogits() { return logits_.Data(); }
    TokenId ArgmaxLogits(int row, int candidate_count);
    void CopyLogits(int row, std::span<float> out);

    void Forward(std::span<const TokenId> tokens, int start_position, LogitsRows logits_rows = LogitsRows::kLast,
                 ForwardObserver* observer = nullptr);
    TokenId ArgmaxLastLogits(int candidate_count) { return ArgmaxLogits(0, candidate_count); }
    void CopyLastLogits(std::span<float> out) { CopyLogits(0, out); }

    void SetMultiRowMath(MultiRowMath math);
    void SetInputStatistics(InputStatistics* statistics);
    void RecordDecodeGraphs();

    void ReplaceWeights(ModelWeights weights);
    void CheckKvCache(std::size_t bytes, KvFormat format) const;
    void ResizeKvCache(std::size_t bytes, KvFormat format);
    bool HasKvCache() const { return cache_.has_value(); }

    KvCacheManager& KvCache() { return *kv_manager_; }
    cudaStream_t Stream() const { return stream_.Get(); }
    const ModelConfig& Config() const { return config_; }
    const TransformerOptions& Options() const { return options_; }
    const ModelWeights& Weights() const { return weights_; }
    std::size_t WeightBytes() const { return weights_.SizeBytes(); }
    std::size_t KvBytesPerToken() const;
    std::size_t DecodeBytes(int context) const;

private:
    struct InputLayout {
        int row_tokens;      // per row: token id
        int row_positions;   // per row: position in its sequence
        int row_cache;       // per row: KV-cache row its keys and values go to
        int row_sequences;   // per row: its sequence's page-table row
        int contexts;        // per sequence: context length (decode attention)
        int logit_rows;      // per logits row: which step row it comes from
        int page_tables;     // per sequence: page table, page_table_stride entries
        int total;
    };

    // What one step runs, as staged in the inputs.
    struct StepShape {
        int rows = 0;              // tokens, all sequences together
        int decode_rows = 0;       // the first rows: sequences with one new token each
        int sequences = 0;         // page-table rows: one per decode row, one per prompt chunk
        int logit_count = 0;       // logits rows to compute
        int first_logit_row = 0;   // their step row, when there is just one
        bool contiguous_logits = true;   // logits row i is step row i
        int prefill_context = 0;   // longest context among the multi-token sequences
    };

    void RunStep(std::span<const StepEntry> entries, ForwardObserver* observer, bool all_logits);
    void ValidateEntries(std::span<const StepEntry> entries) const;
    int MaxSlots() const { return options_.max_batch + options_.max_logit_rows; }
    StepShape StageInputs(std::span<const StepEntry> entries);
    void CopyInputs(int sequences);
    static StepShape DecodeShape(int sequences);
    std::vector<CudaGraph> RecordDecodeStep(const StepShape& shape);
    void EnqueueStep(const StepShape& shape, ForwardObserver* observer, bool all_logits);
    void EnqueueEmbedding(const StepShape& shape, ForwardObserver* observer);
    void EnqueueLogits(const StepShape& shape, ForwardObserver* observer, bool all_logits);
    void ForwardLayer(int layer, const StepShape& shape, ForwardObserver* observer);
    void Observe(ForwardObserver* observer, int layer, std::string_view name, const float* data, int rows, int cols,
                 int row_stride = 0);
    const int* DeviceInputs(int offset) const { return inputs_.Data() + offset; }
    void DropGraphs();

    ModelConfig config_;
    TransformerOptions options_;
    ModelWeights weights_;
    CudaStream stream_;
    Linear linear_;
    // In optionals so ResizeKvCache can free the old cache before allocating
    // the new one, keeping the manager at the same address.
    std::optional<KvCacheManager> kv_manager_;
    std::optional<PagedKvCache> cache_;
    DecodeAttentionWorkspace decode_attention_;
    DeviceBuffer<float> rope_frequencies_;
    KvSequence solo_;   // Forward's sequence

    // Every per-step input lives in one pinned buffer and its GPU mirror,
    // copied over at the start of each step. A decode step reads nothing
    // else from the host, so one recorded graph per batch size replays any
    // step of that size.
    int page_table_stride_;
    int page_shift_;
    InputLayout layout_;
    PinnedBuffer<int> inputs_host_;
    DeviceBuffer<int> inputs_;
    std::vector<std::vector<CudaGraph>> decode_graphs_;   // index: decode rows; each step in pieces
    bool recording_statistics_ = false;      // graphs would replay without the recording, so none are used

    // Activations for up to max_chunk rows, allocated once.
    DeviceBuffer<float> hidden_;        // the residual stream
    DeviceBuffer<float> qkv_;           // stacked q, k, v projections
    DeviceBuffer<float> q_;             // queries after RoPE
    DeviceBuffer<float> attention_;     // attention output, before o_proj
    DeviceBuffer<float> activation_;    // SiLU(gate) * up
    DeviceBuffer<float> logit_input_;   // rows gathered for the vocabulary projection
    DeviceBuffer<float> logits_;        // max_logit_rows x vocab
    DeviceBuffer<float> all_logits_;    // max_chunk x vocab, allocated on first use
    DeviceBuffer<TokenId> argmax_;
};

}  // namespace murmur
