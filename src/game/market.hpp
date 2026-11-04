#pragma once

// The market's NPCs: where they stand, what they're saying, and the idle
// chatter that keeps the engine busy while the player walks around. Each NPC
// has one conversation with the engine, so it remembers both its chatter and
// what the player told it. Every reply ends with an action
// (game/action_language.hpp) that the market carries out: NPCs walk to
// storefronts, hack their signs, wave each other over, and give, sell or
// hand the player things to deliver.

#include "engine/inference_service.hpp"
#include "game/action_language.hpp"
#include "game/market_tuning.hpp"
#include "game/personas.hpp"
#include "game/street.hpp"

#include "raylib.h"

#include <array>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace murmur::game {

struct Npc {
    const personas::Persona* persona = nullptr;
    Vector2 home{};              // their stall
    Vector2 position{};
    Color color{};
    ConversationId conversation = kNoConversation;
    std::string line;            // what they're saying now, or said last
    std::string drafted;         // per byte of line: 1 if the draft model guessed it (speculative decoding)
    float line_age = 0.0f;       // seconds since that line finished
    // The engine is generating their words. One reply at a time: every event
    // for the conversation belongs to it, until its kReplyFinished.
    bool replying = false;
    bool reply_for_player = false;   // that reply answers the player (only then do things change hands)
    bool player_waiting = false;     // the player's message waits for that reply to stop
    ReplyStats last_reply{};     // stats of their last finished reply
    float next_chatter = 0.0f;   // seconds until they call out to passers-by
    std::string action;          // the last action they carried out
    std::string action_forced;   // per byte of it: 1 if the grammar wrote it, not the model
    float action_age = 1e9f;     // seconds since it
    std::string note;            // what happened to them since they last heard from the player
    // An errand: after `errand_delay` seconds, walk to `errand`, stay
    // `errand_stay` seconds, walk home.
    std::optional<Vector2> errand;
    float errand_delay = 0.0f;
    float errand_stay = 0.0f;
    float rested = 1e9f;         // seconds since its last errand ended

    // Busy with the player: answering them, or about to.
    bool TalkingToPlayer() const { return reply_for_player || player_waiting; }
};

// Something an NPC handed the player to take to another NPC.
struct Delivery {
    std::string item;
    std::string from;
    std::string to;
};

struct Offer {
    const Npc* seller = nullptr;
    std::string item;
    int price = 0;
};

struct PlayerState {
    int credits = 60;
    std::vector<std::string> items;
    std::vector<Delivery> deliveries;
    int delivered = 0;
};

enum class BuyResult { kBought, kNoOffer, kTooFewCredits };

class Market {
public:
    explicit Market(InferenceService& service);

    void Update(float delta_time, bool engine_ready, const Npc* talking_to_player);
    void OnEvent(const InferenceEvent& event);
    void DrawStreet() const;
    void Draw(const Font& font, const Npc* talking_to_player) const;

    Npc* NearestTo(Vector2 point, float range);
    void PlayerSays(Npc& npc, std::string message);
    void PlayerLeaves(Npc& npc);
    BuyResult Buy();

    std::span<const Npc> Npcs() const { return npcs_; }
    Npc* FindByName(std::string_view name);
    double LastPlayerFirstWordMs() const { return last_player_first_word_ms_; }
    const Npc* FindByConversation(ConversationId conversation) const;
    void SetChatter(bool enabled) { chatter_ = enabled; }
    const PlayerState& Player() const { return player_; }
    const std::optional<Offer>& CurrentOffer() const { return offer_; }
    int ActionCount(actions::Verb verb) const { return action_counts_[static_cast<std::size_t>(verb)]; }

private:
    Npc* Find(ConversationId conversation);
    void Chatter(Npc& npc);
    void Ask(Npc& npc, std::string message, bool for_player);
    float NextChatterDelay();
    bool Act(Npc& npc, const actions::Action& action);
    void Move(Npc& npc, float delta_time);

    struct PendingMessage {
        Npc* npc;
        std::string text;
    };

    InferenceService* service_;
    std::vector<Npc> npcs_;
    std::vector<SignHack> hacks_;   // one per storefront
    float time_ = 0.0f;
    std::mt19937 random_{ 2026 };
    std::optional<PendingMessage> pending_;   // waits for the NPC's reply in progress to stop
    bool chatter_ = true;
    int cue_ = 0;
    double last_player_first_word_ms_ = 0.0;   // time to first token of the player's latest reply
    PlayerState player_;
    std::optional<Offer> offer_;
    std::array<int, actions::kVerbCount> action_counts_{};   // per actions::Verb
};

}  // namespace murmur::game
