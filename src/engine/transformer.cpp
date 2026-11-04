#include "engine/transformer.hpp"

#include "common/cuda_check.hpp"
#include "engine/kernels/argmax.cuh"
#include "engine/kernels/attention.cuh"
#include "engine/kernels/elementwise.cuh"
#include "engine/kernels/quantize.cuh"
#include "engine/kernels/rope.cuh"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace murmur {

namespace {

constexpr int kNoLayer = -1;
constexpr int kLayersPerGraph = 4;   // layers per recorded piece of a decode step

std::size_t Floats(int rows, int cols) {
    return static_cast<std::size_t>(rows) * cols;
}

int PagesFor(int length, int page_size) {
    return (length + page_size - 1) / page_size;
}

TransformerOptions CheckedOptions(const ModelConfig& config, TransformerOptions options) {
    if (options.max_chunk <= 0 || options.max_context <= 0 || options.max_batch <= 0 ||
        options.max_batch > options.max_chunk) {
        throw std::invalid_argument("Transformer: max_chunk, max_context and max_batch must be positive, "
                                    "and max_batch at most max_chunk");
    }
    if (options.max_logit_rows < options.max_batch || options.max_logit_rows > options.max_chunk) {
        throw std::invalid_argument("Transformer: max_logit_rows must be between max_batch and max_chunk");
    }
    if (options.page_size <= 0 || !std::has_single_bit(static_cast<unsigned>(options.page_size))) {
        throw std::invalid_argument("Transformer: page_size must be a power of two");
    }
    if (options.max_context > config.max_position_embeddings ||
        options.max_context > MaxAttentionContext(config.head_dim)) {
        throw std::invalid_argument("Transformer: max_context exceeds what the model or attention kernel supports");
    }
    return options;
}

// Pages that fit the KV-cache budget; at least one full-length sequence must fit.
int KvPageCount(const ModelConfig& config, const TransformerOptions& options) {
    const std::size_t pages =
        options.kv_cache_bytes / PagedKvCache::BytesPerPage(config, options.page_size, options.kv_format);
    if (pages < static_cast<std::size_t>(PagesFor(options.max_context, options.page_size)) ||
        pages > static_cast<std::size_t>(std::numeric_limits<int>::max() / options.page_size)) {
        throw std::invalid_argument("Transformer: kv_cache_bytes must hold at least one max_context sequence");
    }
    return static_cast<int>(pages);
}

int QkvWidth(const ModelConfig& config) {
    return config.QueryWidth() + 2 * config.KvWidth();
}

LinearLimits LinearLimitsFor(const ModelConfig& config, const ModelWeights& weights, int max_chunk) {
    return {.max_rows = max_chunk,
            .max_in_features = std::max(config.hidden_size, std::max(config.QueryWidth(), config.intermediate_size)),
            .max_norm_width = config.hidden_size,
            .max_swiglu_rows = 2 * config.intermediate_size,
            .max_matrix_elements = weights.LargestLayerMatrix()};
}

}  // namespace

// Takes ownership of the uploaded weights and allocates everything a forward
// pass needs, once: the paged KV cache (as many pages as kv_cache_bytes
// holds), activations for max_chunk tokens and per-step inputs for
// max_batch sequences. Nothing is allocated per token afterwards.
Transformer::Transformer(const ModelConfig& config, ModelWeights weights, TransformerOptions options)
    : config_(config),
      options_(CheckedOptions(config, options)),
      weights_(std::move(weights)),
      linear_(stream_.Get(), LinearLimitsFor(config, weights_, options_.max_chunk), options_.multi_row_math),
      kv_manager_(std::in_place, KvPageCount(config, options_), options_.page_size),
      cache_(std::in_place, config, kv_manager_->PageCount(), options_.page_size, options_.kv_format),
      decode_attention_({.num_heads = config.num_heads,
                         .num_kv_heads = config.num_kv_heads,
                         .head_dim = config.head_dim,
                         .max_context = options_.max_context,
                         .max_batch = options_.max_batch + options_.max_logit_rows}),
      page_table_stride_(PagesFor(options_.max_context, options_.page_size)),
      page_shift_(std::countr_zero(static_cast<unsigned>(options_.page_size))),
      decode_graphs_(options_.max_batch + options_.max_logit_rows + 1),
      hidden_(Floats(options_.max_chunk, config.hidden_size)),
      qkv_(Floats(options_.max_chunk, QkvWidth(config))),
      q_(Floats(options_.max_chunk, config.QueryWidth())),
      attention_(Floats(options_.max_chunk, config.QueryWidth())),
      activation_(Floats(options_.max_chunk, config.intermediate_size)),
      logit_input_(Floats(options_.max_logit_rows, config.hidden_size)),
      logits_(Floats(options_.max_logit_rows, config.vocab_size)),
      argmax_(1) {
    const int chunk = options_.max_chunk;
    const int slots = MaxSlots();
    layout_.row_tokens = 0;
    layout_.row_positions = chunk;
    layout_.row_cache = 2 * chunk;
    layout_.row_sequences = 3 * chunk;
    layout_.contexts = 4 * chunk;
    layout_.logit_rows = layout_.contexts + slots;
    layout_.page_tables = layout_.logit_rows + options_.max_logit_rows;
    layout_.total = layout_.page_tables + slots * page_table_stride_;
    inputs_host_ = PinnedBuffer<int>(layout_.total);
    inputs_ = DeviceBuffer<int>(layout_.total);

    const std::vector<float> frequencies = RopeInverseFrequencies(config.head_dim, config.rope_theta);
    rope_frequencies_ = DeviceBuffer<float>(frequencies.size());
    rope_frequencies_.CopyFromHost(frequencies);
}

// Runs one forward pass over several sequences at once: each entry's new
// tokens are processed at positions entry.sequence->length.., their keys and
// values written into the sequence's pages, and next-token logits computed
// after each of every entry's last entry.logits tokens (StepLogits() holds
// them in entry order, each entry's in token order). All sequences share the
// weight reads: that is what makes batching pay off.
//
// Throws std::invalid_argument for an empty or oversized batch, a sequence
// listed twice, an invalid token id, or pages that don't cover the tokens.
void Transformer::Step(std::span<const StepEntry> entries, ForwardObserver* observer) {
    RunStep(entries, observer, false);
}

// Runs `tokens`, at positions start_position.., through the model on a
// private sequence: their keys and values are cached and the logits of the
// last token (or of every token, for kAll) are computed. A start_position
// before the end of what's cached rewinds and overwrites from there. Prompts
// longer than max_chunk run in several passes; attention reads earlier
// chunks from the cache, so the result is the same as one big pass.
//
// Throws std::invalid_argument for an empty input, an id outside the
// embedding table, a gap in the sequence or a sequence that would overflow
// the context window, and std::runtime_error if the KV cache is full.
void Transformer::Forward(std::span<const TokenId> tokens, int start_position, LogitsRows logits_rows,
                          ForwardObserver* observer) {
    if (tokens.empty()) {
        throw std::invalid_argument("Transformer::Forward: no tokens");
    }
    if (start_position < 0 || start_position > solo_.length ||
        tokens.size() > static_cast<std::size_t>(options_.max_context - start_position)) {
        throw std::invalid_argument("Transformer::Forward: tokens outside the cached sequence or context window");
    }
    if (logits_rows == LogitsRows::kAll && tokens.size() > static_cast<std::size_t>(options_.max_chunk)) {
        throw std::invalid_argument("Transformer::Forward: all-row logits are limited to one chunk");
    }
    if (!std::ranges::all_of(tokens, [&](TokenId id) { return id >= 0 && id < config_.vocab_size; })) {
        throw std::invalid_argument("Transformer::Forward: token id outside the vocabulary");
    }
    if (start_position < solo_.length) {
        kv_manager_->Truncate(solo_, start_position);
    }
    if (!kv_manager_->Reserve(solo_, start_position + static_cast<int>(tokens.size()))) {
        throw std::runtime_error("Transformer::Forward: the KV cache is full");
    }
    const std::size_t chunk = static_cast<std::size_t>(options_.max_chunk);
    for (std::size_t begin = 0; begin < tokens.size(); begin += chunk) {
        const std::span<const TokenId> part = tokens.subspan(begin, std::min(chunk, tokens.size() - begin));
        const bool last_part = begin + part.size() == tokens.size();
        const StepEntry entry{.sequence = &solo_, .tokens = part, .logits = last_part ? 1 : 0};
        RunStep(std::span(&entry, 1), last_part ? observer : nullptr, last_part && logits_rows == LogitsRows::kAll);
        solo_.length += static_cast<int>(part.size());
    }
}

// Stages the step's inputs, then either replays the recorded graph for a
// decode step of this batch size (recording it the first time) or launches
// every kernel directly (prompt chunks, an observer, or all-row logits).
void Transformer::RunStep(std::span<const StepEntry> entries, ForwardObserver* observer, bool all_logits) {
    ValidateEntries(entries);
    // The previous step's input copy may still be reading the pinned buffer.
    stream_.Synchronize();
    const StepShape shape = StageInputs(entries);
    const bool pure_decode = shape.decode_rows == shape.rows && shape.logit_count == shape.rows &&
                             shape.contiguous_logits;
    if (options_.use_cuda_graphs && pure_decode && observer == nullptr && !all_logits && !recording_statistics_) {
        std::vector<CudaGraph>& pieces = decode_graphs_[shape.sequences];
        if (pieces.empty()) {
            pieces = RecordDecodeStep(shape);
        }
        for (const CudaGraph& piece : pieces) {
            piece.Launch(stream_.Get());
            stream_.Flush();
        }
        return;
    }
    CopyInputs(shape.sequences);
    EnqueueStep(shape, observer, all_logits);
}

// Records the decode step of every batch size up to max_batch now, so that
// no step pays for recording one later (a few milliseconds each, while the
// GPU waits). Sizes already recorded are kept.
void Transformer::RecordDecodeGraphs() {
    if (!options_.use_cuda_graphs || recording_statistics_) {
        return;
    }
    stream_.Synchronize();
    for (int sequences = 1; sequences <= options_.max_batch; ++sequences) {
        if (decode_graphs_[sequences].empty()) {
            decode_graphs_[sequences] = RecordDecodeStep(DecodeShape(sequences));
        }
    }
}

// What StageInputs makes of `sequences` entries of one token each.
Transformer::StepShape Transformer::DecodeShape(int sequences) {
    return {.rows = sequences, .decode_rows = sequences, .sequences = sequences, .logit_count = sequences};
}

// Records a pure decode step as a few graphs of kLayersPerGraph layers
// each (the first also copies the inputs and looks up the embeddings, the
// last computes the logits). Launching a graph costs the CPU about a
// microsecond per kernel, and the GPU starts only once the launch returns:
// in pieces, it starts on the first layers while the CPU launches the rest.
std::vector<CudaGraph> Transformer::RecordDecodeStep(const StepShape& shape) {
    std::vector<CudaGraph> pieces;
    for (int begin = 0; begin < config_.num_layers; begin += kLayersPerGraph) {
        const int end = std::min(begin + kLayersPerGraph, config_.num_layers);
        pieces.push_back(CudaGraph::Capture(stream_.Get(), [&] {
            if (begin == 0) {
                CopyInputs(shape.sequences);
                EnqueueEmbedding(shape, nullptr);
            }
            for (int layer = begin; layer < end; ++layer) {
                ForwardLayer(layer, shape, nullptr);
            }
            if (end == config_.num_layers) {
                EnqueueLogits(shape, nullptr, false);
            }
        }));
    }
    return pieces;
}

void Transformer::ValidateEntries(std::span<const StepEntry> entries) const {
    if (entries.empty() || entries.size() > static_cast<std::size_t>(options_.max_batch)) {
        throw std::invalid_argument("Transformer::Step: between 1 and max_batch sequences per step");
    }
    std::size_t rows = 0;
    std::size_t logit_rows = 0;
    std::unordered_set<const KvSequence*> seen;
    for (const StepEntry& entry : entries) {
        if (entry.sequence == nullptr || entry.tokens.empty() || !seen.insert(entry.sequence).second) {
            throw std::invalid_argument("Transformer::Step: each entry needs its own sequence and some tokens");
        }
        const KvSequence& sequence = *entry.sequence;
        const std::size_t end = static_cast<std::size_t>(sequence.length) + entry.tokens.size();
        if (sequence.length < 0 || end > static_cast<std::size_t>(options_.max_context)) {
            throw std::invalid_argument("Transformer::Step: sequence exceeds the context window");
        }
        const int needed_pages = PagesFor(static_cast<int>(end), options_.page_size);
        if (static_cast<int>(sequence.pages.size()) < needed_pages ||
            !std::all_of(sequence.pages.begin(), sequence.pages.begin() + needed_pages,
                         [&](PageId page) { return page >= 0 && page < cache_->PageCount(); })) {
            throw std::invalid_argument("Transformer::Step: the sequence's pages don't cover its tokens");
        }
        if (!std::ranges::all_of(entry.tokens, [&](TokenId id) { return id >= 0 && id < config_.vocab_size; })) {
            throw std::invalid_argument("Transformer::Step: token id outside the vocabulary");
        }
        if (entry.logits < 0 || static_cast<std::size_t>(entry.logits) > entry.tokens.size() ||
            (entry.decode_each_token && static_cast<std::size_t>(entry.logits) != entry.tokens.size())) {
            throw std::invalid_argument("Transformer::Step: logits for more tokens than the entry has, or not for "
                                        "every token decoded one by one");
        }
        rows += entry.tokens.size();
        logit_rows += static_cast<std::size_t>(entry.logits);
    }
    if (rows > static_cast<std::size_t>(options_.max_chunk)) {
        throw std::invalid_argument("Transformer::Step: more tokens than max_chunk");
    }
    if (logit_rows > static_cast<std::size_t>(options_.max_logit_rows)) {
        throw std::invalid_argument("Transformer::Step: more logits rows than max_logit_rows");
    }
}

// Writes every input of the step into the pinned buffer. Each entry gets
// page-table rows ("slots"): one for a prompt chunk, one per token for a
// decode row. Decode rows go first, so the step's first rows are the ones the
// split-K decode attention handles; slot s's page table is row s, and every
// decode row of one sequence shares its pages but sees its own context.
Transformer::StepShape Transformer::StageInputs(std::span<const StepEntry> entries) {
    struct Slot {
        std::size_t entry;
        int first;   // its first token within the entry
        int count;
    };
    std::vector<Slot> slots;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const int count = static_cast<int>(entries[i].tokens.size());
        if (count == 1 || entries[i].decode_each_token) {
            for (int t = 0; t < count; ++t) {
                slots.push_back({i, t, 1});
            }
        }
    }
    StepShape shape;
    shape.decode_rows = static_cast<int>(slots.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].tokens.size() > 1 && !entries[i].decode_each_token) {
            slots.push_back({i, 0, static_cast<int>(entries[i].tokens.size())});
        }
    }
    if (static_cast<int>(slots.size()) > MaxSlots()) {
        throw std::invalid_argument("Transformer::Step: more decode rows and prompt chunks than fit one step");
    }

    int* host = inputs_host_.Data();
    std::vector<int> first_row(entries.size());
    const int mask = options_.page_size - 1;
    for (std::size_t s = 0; s < slots.size(); ++s) {
        const Slot& slot = slots[s];
        const StepEntry& entry = entries[slot.entry];
        const KvSequence& sequence = *entry.sequence;
        const int start = sequence.length + slot.first;
        const int end = start + slot.count;
        std::copy_n(sequence.pages.begin(), PagesFor(end, options_.page_size),
                    host + layout_.page_tables + s * page_table_stride_);
        host[layout_.contexts + s] = end;
        if (slot.first == 0) {
            first_row[slot.entry] = shape.rows;
        }
        for (int t = 0; t < slot.count; ++t) {
            const int row = shape.rows++;
            const int position = start + t;
            host[layout_.row_tokens + row] = entry.tokens[slot.first + t];
            host[layout_.row_positions + row] = position;
            host[layout_.row_cache + row] = (sequence.pages[position >> page_shift_] << page_shift_) | (position & mask);
            host[layout_.row_sequences + row] = static_cast<int>(s);
        }
        if (slot.count > 1) {
            shape.prefill_context = std::max(shape.prefill_context, end);
        }
    }
    shape.sequences = static_cast<int>(slots.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const int last_row = first_row[i] + static_cast<int>(entries[i].tokens.size()) - 1;
        for (int k = entries[i].logits - 1; k >= 0; --k) {
            const int row = last_row - k;
            host[layout_.logit_rows + shape.logit_count] = row;
            shape.contiguous_logits = shape.contiguous_logits && row == shape.logit_count;
            shape.first_logit_row = shape.logit_count == 0 ? row : shape.first_logit_row;
            ++shape.logit_count;
        }
    }
    return shape;
}

// Copies the staged inputs to the GPU: the per-row and per-sequence arrays,
// and the page tables of the step's sequences.
void Transformer::CopyInputs(int sequences) {
    CudaCheck(cudaMemcpyAsync(inputs_.Data(), inputs_host_.Data(), layout_.page_tables * sizeof(int),
                              cudaMemcpyHostToDevice, stream_.Get()));
    CudaCheck(cudaMemcpyAsync(inputs_.Data() + layout_.page_tables, inputs_host_.Data() + layout_.page_tables,
                              static_cast<std::size_t>(sequences) * page_table_stride_ * sizeof(int),
                              cudaMemcpyHostToDevice, stream_.Get()));
}

// Queues the whole forward pass for the staged step: embedding, every
// layer, then the vocabulary projection for the rows that want logits.
// Every kLayersPerGraph layers the queued work is handed to the GPU (see
// CudaStream::Flush), so it runs while the CPU launches the rest.
void Transformer::EnqueueStep(const StepShape& shape, ForwardObserver* observer, bool all_logits) {
    EnqueueEmbedding(shape, observer);
    for (int layer = 0; layer < config_.num_layers; ++layer) {
        ForwardLayer(layer, shape, observer);
        if ((layer + 1) % kLayersPerGraph == 0) {
            stream_.Flush();
        }
    }
    EnqueueLogits(shape, observer, all_logits);
    stream_.Flush();
}

// The embedding lookup: row `token` of the table, for every row of the step.
void Transformer::EnqueueEmbedding(const StepShape& shape, ForwardObserver* observer) {
    DequantizeRows(weights_.embed_tokens.View(), DeviceInputs(layout_.row_tokens), shape.rows, hidden_.Data(),
                   stream_.Get());
    Observe(observer, kNoLayer, "embed", hidden_.Data(), shape.rows, config_.hidden_size);
}

// The vocabulary projection for the rows that want logits (every row with
// all_logits); the final RMSNorm runs inside it.
void Transformer::EnqueueLogits(const StepShape& shape, ForwardObserver* observer, bool all_logits) {
    const cudaStream_t stream = stream_.Get();
    const int hidden = config_.hidden_size;
    const MatVecOptions final_norm{.norm_weight = weights_.final_norm.Data(), .norm_epsilon = config_.rms_norm_eps};
    if (all_logits) {
        if (all_logits_.Size() == 0) {
            all_logits_ = DeviceBuffer<float>(Floats(options_.max_chunk, config_.vocab_size));
        }
        linear_.Apply(hidden_.Data(), weights_.LmHead(), all_logits_.Data(), shape.rows, final_norm);
        Observe(observer, kNoLayer, "logits", all_logits_.Data(), shape.rows, config_.vocab_size);
        CudaCheck(cudaMemcpyAsync(logits_.Data(), all_logits_.Data() + Floats(shape.rows - 1, config_.vocab_size),
                                  Floats(1, config_.vocab_size) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return;
    }
    if (shape.logit_count == 0) {
        return;
    }
    // Only the rows that continue a sequence need next-token scores.
    const float* x = hidden_.Data();
    if (shape.logit_count == 1) {
        x += Floats(shape.first_logit_row, hidden);
    } else if (!shape.contiguous_logits) {
        GatherRows(hidden_.Data(), DeviceInputs(layout_.logit_rows), shape.logit_count, hidden, logit_input_.Data(),
                   stream);
        x = logit_input_.Data();
    }
    linear_.Apply(x, weights_.LmHead(), logits_.Data(), shape.logit_count, final_norm);
}

// One decoder layer on the step's rows, as six fused steps:
//   qkv = (RMSNorm(x)) . Wqkv^T + b              (norm, three projections, bias)
//   q = rope(q); cache[k] = rope(k); cache[v] = v (one kernel, through the page tables)
//   a = attention(q, cache)                      (split-K for decode rows, per row for prompt rows)
//   x += a . Wo^T                                (residual add in the epilogue)
//   h = SiLU(n . Wgate^T) * (n . Wup^T)           with n = RMSNorm(x)
//   x += h . Wdown^T
void Transformer::ForwardLayer(int layer, const StepShape& shape, ForwardObserver* observer) {
    const cudaStream_t stream = stream_.Get();
    const LayerWeights& w = weights_.layers[layer];
    const int rows = shape.rows;
    const int hidden = config_.hidden_size;
    const int q_width = config_.QueryWidth();
    const int kv_width = config_.KvWidth();
    const int qkv_width = QkvWidth(config_);
    const float epsilon = config_.rms_norm_eps;

    linear_.Apply(hidden_.Data(), w.qkv_proj, qkv_.Data(), rows,
                  {.norm_weight = w.attn_norm.Data(), .norm_epsilon = epsilon, .bias = w.qkv_bias.Data()});
    Observe(observer, layer, "q", qkv_.Data(), rows, q_width, qkv_width);
    Observe(observer, layer, "k", qkv_.Data() + q_width, rows, kv_width, qkv_width);
    Observe(observer, layer, "v", qkv_.Data() + q_width + kv_width, rows, kv_width, qkv_width);

    const KvLayer cache = cache_->Layer(layer);
    RopeAndCacheQkv(qkv_.Data(), rows, DeviceInputs(layout_.row_positions), DeviceInputs(layout_.row_cache),
                    rope_frequencies_.Data(),
                    {.num_heads = config_.num_heads, .num_kv_heads = config_.num_kv_heads, .head_dim = config_.head_dim},
                    q_.Data(), cache, stream);

    const PagedKv kv{.page_tables = DeviceInputs(layout_.page_tables),
                     .page_table_stride = page_table_stride_,
                     .page_shift = page_shift_};
    DecodeAttention(q_.Data(), cache, kv, DeviceInputs(layout_.contexts),
                    shape.decode_rows, attention_.Data(), decode_attention_, stream);
    if (rows > shape.decode_rows) {
        const std::size_t offset = Floats(shape.decode_rows, q_width);
        const QueryRows query_rows{.positions = DeviceInputs(layout_.row_positions) + shape.decode_rows,
                                   .sequences = DeviceInputs(layout_.row_sequences) + shape.decode_rows};
        const AttentionShape heads{
            .num_heads = config_.num_heads, .num_kv_heads = config_.num_kv_heads, .head_dim = config_.head_dim};
        // Prompt rows on tensor cores (FP16 inputs), except in exact mode,
        // which keeps every step in FP32 to match the reference.
        if (options_.multi_row_math == MultiRowMath::kExactFp32) {
            CausalAttention(q_.Data() + offset, cache, kv, query_rows, rows - shape.decode_rows,
                            shape.prefill_context, attention_.Data() + offset, heads, stream);
        } else {
            FlashAttention(q_.Data() + offset, cache, kv, query_rows, rows - shape.decode_rows,
                           attention_.Data() + offset, heads, stream);
        }
    }
    Observe(observer, layer, "attn_context", attention_.Data(), rows, q_width);

    linear_.Apply(attention_.Data(), w.o_proj, hidden_.Data(), rows, {.accumulate = true});
    linear_.FeedForward(hidden_.Data(), w.gate_up_proj, w.down_proj, activation_.Data(), rows, w.mlp_norm.Data(),
                        epsilon);
    Observe(observer, layer, "mlp_act", activation_.Data(), rows, config_.intermediate_size);
    Observe(observer, layer, "out", hidden_.Data(), rows, hidden);
}

// Switches how multi-row matrix products are computed (see MultiRowMath).
// Recorded decode graphs bake in the old choice, so they are recorded again.
void Transformer::SetMultiRowMath(MultiRowMath math) {
    stream_.Synchronize();
    linear_.SetMultiRowMath(math);
    options_.multi_row_math = math;
    DropGraphs();
}

// Swaps in other weights for the same model (another WeightFormat, say).
// Recorded graphs read the old matrices' addresses, so they are recorded
// again, and the prefix cache's keys and values were computed with the old
// weights, so it is emptied. Sequences that still hold pages keep theirs;
// the caller evicts them (Engine::Reconfigure does).
void Transformer::ReplaceWeights(ModelWeights weights) {
    if (weights.layers.size() != weights_.layers.size() ||
        weights.LargestLayerMatrix() != weights_.LargestLayerMatrix()) {
        throw std::invalid_argument("Transformer::ReplaceWeights: weights of a different model");
    }
    stream_.Synchronize();
    weights_ = std::move(weights);
    kv_manager_->Release(solo_);
    kv_manager_->ForgetCachedPrefixes();
    DropGraphs();
}

// Throws std::invalid_argument if a KV cache of `bytes` in `format` can't
// serve this transformer (it must hold at least one max_context sequence).
void Transformer::CheckKvCache(std::size_t bytes, KvFormat format) const {
    TransformerOptions next = options_;
    next.kv_cache_bytes = bytes;
    next.kv_format = format;
    static_cast<void>(KvPageCount(config_, next));
}

// Replaces the KV cache with one of `bytes` in `format` (more or fewer
// pages). Every page dies with the old cache, so no sequence may still hold
// one: the private sequence of Forward and the prefix cache are emptied
// here, and std::logic_error reports any other holder. The old cache is
// freed before the new one is allocated, so the two never coexist; if the
// new one can't be allocated, a cache of the old size and format is put
// back and the error rethrown. Only if even that fails is the transformer
// left without a cache (HasKvCache), unable to run.
void Transformer::ResizeKvCache(std::size_t bytes, KvFormat format) {
    TransformerOptions next = options_;
    next.kv_cache_bytes = bytes;
    next.kv_format = format;
    const int pages = KvPageCount(config_, next);   // validates before anything is freed
    stream_.Synchronize();
    kv_manager_->Release(solo_);
    while (kv_manager_->EvictCachedPage()) {
    }
    if (kv_manager_->FreePages() != kv_manager_->PageCount()) {
        throw std::logic_error("Transformer::ResizeKvCache: sequences still hold pages");
    }
    const int old_pages = kv_manager_->PageCount();
    DropGraphs();   // they write into the old cache
    cache_.reset();
    kv_manager_.reset();
    try {
        cache_.emplace(config_, pages, next.page_size, next.kv_format);
    } catch (...) {
        kv_manager_.emplace(old_pages, options_.page_size);
        cache_.emplace(config_, old_pages, options_.page_size, options_.kv_format);
        throw;
    }
    options_ = next;
    kv_manager_.emplace(pages, options_.page_size);
}

void Transformer::DropGraphs() {
    for (std::vector<CudaGraph>& pieces : decode_graphs_) {
        pieces.clear();
    }
}

// Starts (or, with nullptr, stops) recording the inputs of every matrix
// product into `statistics`, for calibration. While recording, every step
// launches its kernels directly.
void Transformer::SetInputStatistics(InputStatistics* statistics) {
    stream_.Synchronize();
    linear_.SetInputStatistics(statistics);
    recording_statistics_ = statistics != nullptr;
}

// GPU memory one cached token takes: keys and values, every layer.
std::size_t Transformer::KvBytesPerToken() const {
    return PagedKvCache::BytesPerPage(config_, 1, options_.kv_format);
}

// Bytes one decode step must read from memory with `context` tokens cached:
// every weight once, plus every cached key and value (the bandwidth floor).
std::size_t Transformer::DecodeBytes(int context) const {
    return WeightBytes() + static_cast<std::size_t>(context) * KvBytesPerToken();
}

// Reports a checkpoint named "<name>" (layer == kNoLayer) or
// "layers.<layer>.<name>". The name is only built when someone is listening.
// A row_stride of 0 means the rows are packed (stride = cols).
void Transformer::Observe(ForwardObserver* observer, int layer, std::string_view name, const float* data, int rows,
                          int cols, int row_stride) {
    if (observer == nullptr) {
        return;
    }
    stream_.Synchronize();
    const std::string full_name =
        layer == kNoLayer ? std::string(name) : "layers." + std::to_string(layer) + "." + std::string(name);
    observer->OnCheckpoint(full_name, data, rows, cols, row_stride == 0 ? cols : row_stride);
}

// The highest-scoring next token of logits row `row` among ids [0,
// candidate_count): pass the tokenizer's vocabulary size so the embedding
// table's padding rows (ids no tokenizer can decode) can never be chosen.
// Blocks until the GPU is done.
TokenId Transformer::ArgmaxLogits(int row, int candidate_count) {
    if (row < 0 || row >= options_.max_logit_rows || candidate_count <= 0 || candidate_count > config_.vocab_size) {
        throw std::invalid_argument("Transformer::ArgmaxLogits: bad row or candidate count");
    }
    Argmax(logits_.Data() + Floats(row, config_.vocab_size), candidate_count, argmax_.Data(), stream_.Get());
    TokenId result = 0;
    CudaCheck(cudaMemcpyAsync(&result, argmax_.Data(), sizeof(result), cudaMemcpyDeviceToHost, stream_.Get()));
    stream_.Synchronize();
    return result;
}

// Copies the first out.size() logits of row `row` to `out` (CPU), for
// sampling on the host. Blocks until the GPU is done.
void Transformer::CopyLogits(int row, std::span<float> out) {
    if (row < 0 || row >= options_.max_logit_rows || out.size() > static_cast<std::size_t>(config_.vocab_size)) {
        throw std::invalid_argument("Transformer::CopyLogits: bad row or more logits than the vocabulary");
    }
    CudaCheck(cudaMemcpyAsync(out.data(), logits_.Data() + Floats(row, config_.vocab_size), out.size_bytes(),
                              cudaMemcpyDeviceToHost, stream_.Get()));
    stream_.Synchronize();
}

}  // namespace murmur
