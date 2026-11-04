#pragma once

// The player talking to the NPC they're standing next to: the chat box they
// type in, the speech bubble the NPC's reply streams into (words the draft
// model guessed in teal), and the NPC's offer, if it made one.

#include "game/market.hpp"

#include "raylib.h"

#include <string>
#include <string_view>

namespace murmur::game {

class Dialogue {
public:
    Dialogue(Market& market, const Font& font);

    void Update(Vector2 player, float delta_time, bool engine_ready);
    void Draw(Vector2 player) const;
    void Say(Npc& npc, std::string message);

    bool CapturesKeyboard() const { return typing_; }
    const Npc* Partner() const { return partner_; }
    bool Replying() const { return partner_ != nullptr && partner_->TalkingToPlayer(); }
    std::string Reply() const { return partner_ != nullptr ? partner_->line : std::string(); }

private:
    void HandleTyping();
    void DrawSpeechBubble() const;
    void DrawReplyLine(std::string_view line, std::size_t first, float x, float y) const;
    void DrawChatBox() const;

    Market* market_;
    const Font* font_;
    Npc* partner_ = nullptr;   // the NPC the player is talking to (or standing next to)
    bool typing_ = false;
    std::string input_;
    std::string player_line_;
    float time_ = 0.0f;   // drives the "thinking" dots and the cursor blink
    std::string refusal_;   // why B didn't buy, shown in place of the hint for a moment
    float refusal_seconds_ = 0.0f;
};

}  // namespace murmur::game
