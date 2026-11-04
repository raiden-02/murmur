#include "game/hud.hpp"

#include "game/street.hpp"

#include <algorithm>
#include <cctype>
#include <format>
#include <string_view>
#include <utility>

namespace murmur::game {

namespace {

constexpr float kTextSize = 18.0f;
constexpr float kSmallText = 15.0f;
constexpr float kSpacing = 1.0f;

constexpr Rectangle kPanel{ 846, kStreetTop + 6, 428, kStreetBottom - kStreetTop - 12 };
constexpr float kMapHeight = 120.0f;   // the page map shrinks its squares to fit any cache in this
constexpr float kMaxCell = 8.0f;

constexpr Color kPanelFill{ 14, 12, 22, 230 };
constexpr Color kPanelEdge{ 93, 202, 165, 255 };
constexpr Color kFreePage{ 38, 36, 48, 255 };
constexpr Color kCachedPage{ 96, 96, 108, 255 };
constexpr Color kHint{ 250, 199, 117, 255 };
constexpr Color kDim{ 170, 168, 180, 255 };
constexpr Color kWeights{ 250, 199, 117, 255 };
constexpr float kFrameBudgetMs = 1000.0f / kTargetFps;
constexpr float kNoticeSeconds = 4.0f;

void Text(const Font& font, const std::string& text, float x, float y, float size, Color color) {
    DrawTextEx(font, text.c_str(), { x, y }, size, kSpacing, color);
}

std::string Upper(std::string_view text) {
    std::string upper(text);
    for (char& c : upper) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return upper;
}

double Gigabytes(std::size_t bytes) {
    return static_cast<double>(bytes) / 1e9;
}

Color PageColor(const Market& market, ConversationId owner) {
    if (owner == EngineStats::kFree) {
        return kFreePage;
    }
    if (owner == EngineStats::kCached) {
        return kCachedPage;
    }
    const Npc* npc = market.FindByConversation(owner);
    return npc != nullptr ? npc->color : RAYWHITE;
}

}  // namespace

// Engine-wide news: loading finished, the engine stopped, or a format switch
// finished or failed. A failed engine-wide request (a format switch, the
// draft model) leaves the engine running as it was, and the player hears
// why in a notice.
void Hud::OnEvent(const InferenceEvent& event) {
    if (event.kind == InferenceEventKind::kReady) {
        state_ = State::kReady;
    } else if (event.kind == InferenceEventKind::kReconfigured) {
        switching_ = false;
    } else if (event.kind == InferenceEventKind::kError && event.conversation == kNoConversation) {
        switching_ = false;
        Notify("engine: " + event.text);
    } else if (event.kind == InferenceEventKind::kStopped && state_ != State::kMissingModel) {
        state_ = State::kFailed;   // a missing model already says more than the loader's error
        error_ = event.text;
    }
}

// Shows `text` above the status line for a few seconds, replacing any
// notice still up.
void Hud::Notify(std::string text) {
    notice_ = std::move(text);
    notice_seconds_ = kNoticeSeconds;
}

// The model's folder lacks `files`: the game can't talk, so instead of a
// loader error it shows what to download and where.
void Hud::ShowMissingModel(const std::filesystem::path& model_dir, const std::vector<std::string>& files) {
    state_ = State::kMissingModel;
    model_dir_ = std::filesystem::absolute(model_dir);
    missing_files_.clear();
    for (const std::string& file : files) {
        missing_files_ += missing_files_.empty() ? file : ", " + file;
    }
    error_ = std::format("the model is missing {} in {}", missing_files_, model_dir_.string());
}

void Hud::Update(const EngineStats& stats, float frame_seconds) {
    notice_seconds_ = std::max(0.0f, notice_seconds_ - frame_seconds);
    const float frame_ms = frame_seconds * 1000.0f;
    frame_ms_ = frame_ms_ == 0.0f ? frame_ms : 0.9f * frame_ms_ + 0.1f * frame_ms;
    worst_frame_window_ = std::max(worst_frame_window_, frame_ms);
    window_seconds_ += frame_seconds;
    if (window_seconds_ >= 0.5) {
        tokens_per_second_ = static_cast<double>(stats.generated_tokens - window_start_tokens_) / window_seconds_;
        window_start_tokens_ = stats.generated_tokens;
        window_seconds_ = 0.0;
        worst_frame_ms_ = worst_frame_window_;
        worst_frame_window_ = 0.0f;
    }
}

void Hud::Draw(const Market& market, const EngineStats& stats) const {
    if (xray_) {
        DrawXRay(market, stats);
    }
    DrawStatus(market, stats);
    DrawPockets(market);
    DrawNotice();
    if (state_ == State::kMissingModel) {
        DrawMissingModel();
    }
}

// The latest notice, centred over the street like a toast, fading out in
// its last second.
void Hud::DrawNotice() const {
    if (notice_seconds_ <= 0.0f) {
        return;
    }
    const float alpha = std::min(1.0f, notice_seconds_);
    const float width = MeasureTextEx(*font_, notice_.c_str(), kTextSize, kSpacing).x;
    const Rectangle box{ (kScreenWidth - width) / 2 - 12, (kStreetTop + kStreetBottom) / 2 - 75, width + 24, 30 };
    DrawRectangleRounded(box, 0.3f, 6, Fade(kPanelFill, alpha));
    Text(*font_, notice_, box.x + 12, box.y + 5, kTextSize, Fade(kHint, alpha));
}

// What the game needs before anyone can talk: which files are missing, the
// folder they belong in, and how to fetch them.
void Hud::DrawMissingModel() const {
    constexpr Rectangle kBox{ 140, kStreetTop + 90, kScreenWidth - 280, 206 };
    DrawRectangleRounded(kBox, 0.05f, 6, kPanelFill);
    DrawRectangleRoundedLinesEx(kBox, 0.05f, 6, 1.5f, kHint);
    const float x = kBox.x + 24;
    float y = kBox.y + 20;
    Text(*font_, "The NPCs need their model before they can talk", x, y, kTextSize + 4, kHint);
    y += 40;
    Text(*font_, "Missing: " + missing_files_, x, y, kTextSize, RAYWHITE);
    y += 26;
    Text(*font_, "in " + model_dir_.string(), x, y, kSmallText, kDim);
    y += 34;
    Text(*font_, "Download Qwen2.5-1.5B-Instruct (about 3.1 GB) into that folder, for example:", x, y, kSmallText,
         RAYWHITE);
    y += 22;
    Text(*font_, "huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir \"" + model_dir_.string() + "\"",
         x + 16, y, kSmallText, kPanelEdge);
    y += 30;
    Text(*font_, "or set MURMUR_MODEL_DIR to a folder that has it. Then start the game again.", x, y, kSmallText,
         RAYWHITE);
}

// Credits, what the player carries, and deliveries still to make, in the
// dark band under the upper storefronts.
void Hud::DrawPockets(const Market& market) const {
    const PlayerState& player = market.Player();
    std::string items;
    for (const std::string& item : player.items) {
        items += items.empty() ? item : ", " + item;
    }
    std::string text = std::format("{} credits    {}", player.credits, items.empty() ? "empty pockets" : items);
    for (const Delivery& delivery : player.deliveries) {
        text += std::format("    deliver {} to {}", delivery.item, delivery.to);
    }
    if (player.delivered > 0) {
        text += std::format("    {} delivered", player.delivered);
    }
    Text(*font_, text, 20, kStreetTop - 26, kSmallText, kHint);
}

// What the NPCs did, and who wrote it: of every action's tokens, the share
// the grammar fixed (jump-forward, no model step needed). Then the draft
// model: of its guesses, the share the model kept. Returns the y below.
float Hud::DrawActions(const Market& market, const EngineStats& stats, float x, float y) const {
    std::string counts;
    for (const actions::Verb verb : { actions::Verb::kWalk, actions::Verb::kHack, actions::Verb::kCall,
             actions::Verb::kGive, actions::Verb::kSell, actions::Verb::kQuest, actions::Verb::kNone }) {
        counts += std::format("{} {}  ", actions::VerbName(verb), market.ActionCount(verb));
    }
    Text(*font_, std::format("actions {}:  {}", stats.actions, counts), x, y, kSmallText, RAYWHITE);
    y += 20;
    const std::int64_t action_tokens = stats.action_tokens + stats.forced_tokens;
    const double forced = action_tokens > 0 ? 100.0 * stats.forced_tokens / action_tokens : 0.0;
    Text(*font_, std::format("grammar wrote {:.0f}% of action tokens itself", forced), x, y, kSmallText, kDim);
    y += 20;
    if (!draft_model_installed_) {
        Text(*font_, "draft model: not installed", x, y, kSmallText, kDim);
    } else if (stats.draft_tokens == 0) {
        Text(*font_, "draft model: off (G)", x, y, kSmallText, kDim);
    } else if (stats.drafted_tokens > 0) {
        Text(*font_, std::format("draft model: {} of {} guesses kept ({:.0f}%)", stats.accepted_tokens,
            stats.drafted_tokens, 100.0 * stats.accepted_tokens / stats.drafted_tokens), x, y, kSmallText, kDim);
    } else {
        Text(*font_, std::format("draft model: waits until {} or fewer are talking", stats.max_speculative_batch), x,
            y, kSmallText, kDim);
    }
    return y + 24;
}

// Controls and the engine's state, in the dark band above the lower storefronts.
void Hud::DrawStatus(const Market& market, const EngineStats& stats) const {
    const float y = kStreetBottom + 4;
    Text(*font_, "WASD walk  E talk  Enter send  B buy  Esc close  Tab X-ray  Q weights  K cache  G draft", 20,
        y + 2, kSmallText, LIGHTGRAY);
    switch (state_) {
    case State::kLoading:
        Text(*font_, "engine: loading model...", 640, y, kTextSize, kHint);
        break;
    case State::kReady: {
        if (switching_) {
            Text(*font_, "engine: reloading the weights...", 640, y, kTextSize, kHint);
            break;
        }
        const auto talking = std::ranges::count_if(market.Npcs(), &Npc::replying);
        Text(*font_, std::format("{} weights, {} cache   {} talking   {:.0f} tokens/s",
            Upper(WeightFormatName(stats.weight_format)), Upper(KvFormatName(stats.kv_format)), talking,
            tokens_per_second_), 640, y, kTextSize, GREEN);
        break;
    }
    case State::kMissingModel:
        Text(*font_, "engine: no model installed", 640, y, kTextSize, RED);
        break;
    case State::kFailed:
        Text(*font_, "engine failed: " + error_, 640, y, kTextSize, RED);
        break;
    }
}

// The model's memory budget as one bar, split in proportion: the weights
// (amber) and the KV cache (teal) that fills the rest. Smaller weights leave
// a longer teal part: room for more NPCs' memories. Returns the y below it.
float Hud::DrawMemory(const EngineStats& stats, float x, float y) const {
    const float width = kPanel.width - 24;
    const double total = static_cast<double>(stats.weight_bytes + stats.kv_bytes);
    const float split = total > 0.0 ? static_cast<float>(width * stats.weight_bytes / total) : 0.0f;
    DrawRectangleRec({ x, y, split, 14 }, kWeights);
    DrawRectangleRec({ x + split, y, width - split, 14 }, kPanelEdge);
    y += 18;
    const int pages = static_cast<int>(stats.page_owner.size());
    Text(*font_, std::format("{} weights {:.2f} GB", Upper(WeightFormatName(stats.weight_format)),
        Gigabytes(stats.weight_bytes)), x, y, kSmallText, kWeights);
    Text(*font_, std::format("{} cache: {} NPC memories", Upper(KvFormatName(stats.kv_format)),
        pages * stats.page_tokens / kNpcMemoryTokens), x + 200, y, kSmallText, kPanelEdge);
    return y + 24;
}

// Each small square is one page of the KV cache (a run of tokens), coloured by the
// NPC whose conversation it holds; outlined squares are shared by several
// conversations (the city lore every NPC starts with), grey ones are cached
// prefixes nobody is using right now, dark ones free. The squares shrink
// until the whole cache fits the map's area, so a bigger cache shows as
// more, smaller squares. Returns the y below the map.
float Hud::DrawPageMap(const Market& market, const EngineStats& stats, float x0, float y) const {
    const int pages = static_cast<int>(stats.page_owner.size());
    const float width = kPanel.width - 24;
    float cell = kMaxCell;
    float gap = 1.0f;
    int columns = 1;
    while (true) {
        columns = static_cast<int>(width / (cell + gap));
        const int rows = (pages + columns - 1) / columns;
        if (rows * (cell + gap) <= kMapHeight || cell <= 1.0f) {
            break;
        }
        cell -= 1.0f;
        gap = cell >= 4.0f ? 1.0f : 0.0f;
    }
    for (int page = 0; page < pages; ++page) {
        const float x = x0 + (page % columns) * (cell + gap);
        const float cy = y + (page / columns) * (cell + gap);
        DrawRectangleRec({ x, cy, cell, cell }, PageColor(market, stats.page_owner[page]));
        if (stats.page_shared[page] != 0 && cell >= 4.0f) {
            DrawRectangleLinesEx({ x - 0.5f, cy - 0.5f, cell + 1, cell + 1 }, 1.0f, RAYWHITE);
        }
    }
    return y + kMapHeight + 6;
}

// The X-ray panel: memory, pages, batch, speed.
void Hud::DrawXRay(const Market& market, const EngineStats& stats) const {
    DrawRectangleRounded(kPanel, 0.03f, 6, kPanelFill);
    DrawRectangleRoundedLinesEx(kPanel, 0.03f, 6, 1.5f, kPanelEdge);
    const float x0 = kPanel.x + 12;
    float y = kPanel.y + 10;
    const int pages = static_cast<int>(stats.page_owner.size());
    const int used = pages - stats.pages_free;
    Text(*font_, "X-RAY  model memory", x0, y, kTextSize, kPanelEdge);
    Text(*font_, std::format("{:.2f} GB", Gigabytes(stats.weight_bytes + stats.kv_bytes)), x0 + 250, y + 2, kSmallText,
        kDim);
    y += 24;
    y = DrawMemory(stats, x0, y);
    Text(*font_, std::format("KV cache: {} / {} pages in use", used, pages), x0, y, kSmallText, RAYWHITE);
    y += 20;
    y = DrawPageMap(market, stats, x0, y);

    const auto legend = [&](float x, Color color, bool outline, const char* label) {
        DrawRectangleRec({ x, y + 3, kMaxCell, kMaxCell }, color);
        if (outline) {
            DrawRectangleLinesEx({ x - 0.5f, y + 2.5f, kMaxCell + 1, kMaxCell + 1 }, 1.0f, RAYWHITE);
        }
        Text(*font_, label, x + 13, y, kSmallText, kDim);
    };
    legend(x0, market.Npcs().front().color, false, "an NPC's");
    legend(x0 + 92, kCachedPage, true, "shared");
    legend(x0 + 172, kCachedPage, false, "cached, idle");
    legend(x0 + 292, kFreePage, false, "free");
    y += 26;

    // The last step's batch: one slot per conversation a step can hold,
    // filled for a decode row, outlined for a prompt chunk.
    Text(*font_, std::format("batch: {} decoding + {} prompt tokens", stats.last_decode_rows, stats.last_prefill_rows),
        x0, y, kSmallText, RAYWHITE);
    y += 22;
    const std::size_t slots = std::max(static_cast<std::size_t>(stats.max_batch), stats.last_batch.size());
    const float pitch = slots > 0 ? (kPanel.width - 24) / static_cast<float>(slots) : 0.0f;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const float x = x0 + slot * pitch;
        const Rectangle box{ x, y, pitch * 0.8f, 14 };
        if (slot < stats.last_batch.size()) {
            const Color color = PageColor(market, stats.last_batch[slot]);
            if (static_cast<int>(slot) < stats.last_decode_rows) {
                DrawRectangleRec(box, color);
            } else {
                DrawRectangleLinesEx(box, 1.0f, color);
            }
        } else {
            DrawRectangleRec(box, kFreePage);
        }
    }
    y += 26;

    Text(*font_, std::format("step {:.1f} ms    {:.0f} tokens/s", stats.last_step_ms, tokens_per_second_), x0, y,
        kSmallText, RAYWHITE);
    y += 20;
    Text(*font_, std::format("evicted {}    prompt tokens reused {}", stats.evictions, stats.reused_tokens), x0, y,
        kSmallText, kDim);
    y += 20;
    if (market.LastPlayerFirstWordMs() > 0.0) {
        Text(*font_, std::format("your last reply: first word after {:.0f} ms", market.LastPlayerFirstWordMs()), x0, y,
            kSmallText, kHint);
    }
    y += 24;
    y = DrawActions(market, stats, x0, y);

    // Frame time against the 60 fps budget: the engine runs on its own thread,
    // so the bar should stay short however busy the market gets.
    Text(*font_, std::format("frame {:.1f} ms (worst {:.1f})", frame_ms_, worst_frame_ms_), x0, y, kSmallText, RAYWHITE);
    y += 20;
    const float bar_width = kPanel.width - 24;
    const float scale = bar_width / (2 * kFrameBudgetMs);
    DrawRectangleRec({ x0, y, bar_width, 10 }, kFreePage);
    DrawRectangleRec({ x0, y, std::min(bar_width, frame_ms_ * scale), 10 },
        frame_ms_ <= kFrameBudgetMs + 0.5f ? kPanelEdge : RED);
    DrawLineV({ x0 + kFrameBudgetMs * scale, y - 3 }, { x0 + kFrameBudgetMs * scale, y + 13 }, kHint);
}

}  // namespace murmur::game
