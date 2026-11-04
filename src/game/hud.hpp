#pragma once

// The status line, the player's credits and pockets, and the X-ray panel
// that shows the engine at work: how the model's memory splits between
// weights and KV cache, the cache's pages by owner, the batch of the last
// step, what the NPCs did (and how much of it the grammar wrote), how the
// draft model's guesses fared, throughput and the frame budget. Also brief
// notices (a key that did nothing, a request the engine refused) and, when
// the model isn't installed, a panel saying what to download and where.

#include "engine/inference_service.hpp"
#include "game/market.hpp"

#include "raylib.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace murmur::game {

class Hud {
public:
    explicit Hud(const Font& font) : font_(&font) {}

    void OnEvent(const InferenceEvent& event);
    void Update(const EngineStats& stats, float frame_seconds);
    void Draw(const Market& market, const EngineStats& stats) const;

    void Notify(std::string text);
    void ShowMissingModel(const std::filesystem::path& model_dir, const std::vector<std::string>& files);
    void SetDraftModelInstalled(bool installed) { draft_model_installed_ = installed; }

    bool EngineReady() const { return state_ == State::kReady; }
    bool EngineFailed() const { return state_ == State::kFailed || state_ == State::kMissingModel; }
    const std::string& Error() const { return error_; }
    bool Switching() const { return switching_; }
    void OnFormatsRequested() { switching_ = true; }
    void ToggleXRay() { xray_ = !xray_; }
    void ShowXRay(bool visible) { xray_ = visible; }

private:
    enum class State { kLoading, kReady, kFailed, kMissingModel };

    void DrawStatus(const Market& market, const EngineStats& stats) const;
    void DrawPockets(const Market& market) const;
    void DrawNotice() const;
    void DrawMissingModel() const;
    float DrawActions(const Market& market, const EngineStats& stats, float x, float y) const;
    void DrawXRay(const Market& market, const EngineStats& stats) const;
    float DrawMemory(const EngineStats& stats, float x, float y) const;
    float DrawPageMap(const Market& market, const EngineStats& stats, float x, float y) const;

    const Font* font_;
    State state_ = State::kLoading;
    std::string error_;          // why the engine failed or can't start
    std::filesystem::path model_dir_;   // kMissingModel: where the model belongs...
    std::string missing_files_;         // ...and what isn't there, comma-separated
    bool draft_model_installed_ = true;
    bool xray_ = false;
    bool switching_ = false;   // a format change is on its way (until kReconfigured)
    std::string notice_;
    float notice_seconds_ = 0.0f;   // how long the notice stays up

    // Throughput over the last half second, and frame times.
    double window_seconds_ = 0.0;
    std::int64_t window_start_tokens_ = 0;
    double tokens_per_second_ = 0.0;
    float frame_ms_ = 0.0f;
    float worst_frame_ms_ = 0.0f;
    float worst_frame_window_ = 0.0f;
};

}  // namespace murmur::game
