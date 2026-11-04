#include "game/dialogue.hpp"

#include "game/street.hpp"
#include "game/text_layout.hpp"
#include "model/unicode.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace murmur::game {

namespace {

constexpr float kTalkDistance = 60.0f;    // close enough to start talking
constexpr float kLeaveDistance = 150.0f;  // walking this far away ends the conversation
constexpr std::size_t kMaxInputBytes = 240;
constexpr float kFontSize = 20.0f;
constexpr float kLetterSpacing = 1.0f;
constexpr float kLineHeight = 24.0f;
constexpr float kBubbleMaxWidth = 420.0f;
constexpr float kBubblePadding = 12.0f;
constexpr int kBubbleMaxLines = 7;
constexpr float kRefusalSeconds = 2.5f;

constexpr Color kBubbleFill{ 245, 240, 255, 240 };
constexpr Color kBubbleText{ 38, 33, 92, 255 };
constexpr Color kDraftedText{ 0, 128, 118, 255 };   // guessed by the draft model, confirmed by the model
constexpr Color kPanel{ 20, 18, 30, 220 };
constexpr Color kHint{ 250, 199, 117, 255 };
constexpr Color kRefusal{ 240, 149, 149, 255 };

float Distance(Vector2 a, Vector2 b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

// raylib takes null-terminated strings, so these copy a view first.
float TextWidth(const Font& font, std::string_view text) {
    return MeasureTextEx(font, std::string(text).c_str(), kFontSize, kLetterSpacing).x;
}

void DrawUiText(const Font& font, std::string_view text, float x, float y, Color color) {
    DrawTextEx(font, std::string(text).c_str(), { x, y }, kFontSize, kLetterSpacing, color);
}

// The end of `text` that fits in `max_width`, so the caret stays visible
// while typing a long message.
std::string Tail(const Font& font, std::string text, float max_width) {
    while (!text.empty() && TextWidth(font, text) > max_width) {
        int bytes = 1;
        GetCodepointNext(text.c_str(), &bytes);
        text.erase(0, static_cast<std::size_t>(bytes));
    }
    return text;
}

}  // namespace

Dialogue::Dialogue(Market& market, const Font& font) : market_(&market), font_(&font) {}

// Once per frame: keep the partner while the player stays near (or pick the
// nearest NPC), then handle typing. Walking away closes the chat box and
// cuts the NPC's reply short. B buys what the partner offers; when the
// player can't pay, the hint says so instead of nothing happening.
void Dialogue::Update(Vector2 player, float delta_time, bool engine_ready) {
    time_ += delta_time;
    refusal_seconds_ = std::max(0.0f, refusal_seconds_ - delta_time);
    const float stay_within = typing_ || Replying() ? kLeaveDistance : kTalkDistance;
    if (partner_ != nullptr && Distance(player, partner_->position) > stay_within) {
        market_->PlayerLeaves(*partner_);
        partner_ = nullptr;
        typing_ = false;
        player_line_.clear();
        refusal_seconds_ = 0.0f;
    }
    if (partner_ == nullptr) {
        partner_ = market_->NearestTo(player, kTalkDistance);
    }
    if (partner_ == nullptr) {
        return;
    }
    if (typing_) {
        HandleTyping();
        return;
    }
    const std::optional<Offer>& offer = market_->CurrentOffer();
    if (offer && offer->seller == partner_ && IsKeyPressed(KEY_B)) {
        const std::string item = offer->item;
        const int price = offer->price;
        if (market_->Buy() == BuyResult::kTooFewCredits) {
            refusal_ = std::format("The {} costs {} credits; you have {}", item, price, market_->Player().credits);
            refusal_seconds_ = kRefusalSeconds;
        }
    }
    if (engine_ready && !Replying() && (IsKeyPressed(KEY_E) || IsKeyPressed(KEY_ENTER))) {
        typing_ = true;
        while (GetCharPressed() != 0) {
            // Discard the 'e' that opened the box.
        }
    }
}

void Dialogue::HandleTyping() {
    for (int c = GetCharPressed(); c != 0; c = GetCharPressed()) {
        if (c >= 32 && input_.size() + 4 <= kMaxInputBytes) {
            unicode::AppendUtf8(static_cast<char32_t>(c), input_);
        }
    }
    if (IsKeyPressed(KEY_BACKSPACE) || IsKeyPressedRepeat(KEY_BACKSPACE)) {
        PopLastCodepoint(input_);
    }
    if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER)) {
        typing_ = false;
        Say(*partner_, std::exchange(input_, {}));
    } else if (IsKeyPressed(KEY_ESCAPE)) {
        typing_ = false;
    }
}

// Sends the player's message to `npc`; the reply streams into the speech
// bubble. Blank messages are ignored.
void Dialogue::Say(Npc& npc, std::string message) {
    if (std::ranges::all_of(message, [](char c) { return c == ' '; })) {
        return;
    }
    partner_ = &npc;
    player_line_ = message;
    market_->PlayerSays(npc, std::move(message));
}

void Dialogue::Draw(Vector2 player) const {
    DrawCircleV(player, kPlayerRadius, RAYWHITE);
    DrawUiText(*font_, "You", player.x - TextWidth(*font_, "You") / 2, player.y + 14, LIGHTGRAY);
    if (partner_ != nullptr) {
        DrawCircleLinesV(partner_->position, kPlayerRadius + 9, kHint);
        if (Replying() || (!player_line_.empty() && !partner_->line.empty())) {
            DrawSpeechBubble();
        }
    }
    DrawChatBox();
}

// The reply so far, word-wrapped, beside the partner, towards the middle of
// the street. Long replies show their most recent lines; before the first
// word arrives, animated dots.
void Dialogue::DrawSpeechBubble() const {
    const std::string thinking(1 + static_cast<int>(time_ * 3.0f) % 3, '.');
    const bool waiting = partner_->player_waiting || partner_->line.empty();
    const std::string& text = waiting ? thinking : partner_->line;
    std::vector<std::string_view> lines = WrapText(*font_, text, kFontSize, kBubbleMaxWidth);
    if (lines.size() > kBubbleMaxLines) {
        lines.erase(lines.begin(), lines.end() - kBubbleMaxLines);
    }
    float width = 0.0f;
    for (const std::string_view line : lines) {
        width = std::max(width, TextWidth(*font_, line));
    }
    const float box_width = width + 2 * kBubblePadding;
    const float box_height = lines.size() * kLineHeight + 2 * kBubblePadding;
    const Vector2 npc = partner_->position;
    const bool top_row = npc.y < (kStreetTop + kStreetBottom) / 2;
    const float x = std::clamp(npc.x - box_width / 2, 8.0f, kScreenWidth - box_width - 8.0f);
    const float y = top_row ? npc.y + 44 : npc.y - 44 - box_height;
    DrawRectangleRounded({ x, y, box_width, box_height }, 0.12f, 6, kBubbleFill);
    DrawRectangleLinesEx({ x, y, 4, box_height }, 4, partner_->color);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const float line_y = y + kBubblePadding + i * kLineHeight;
        if (waiting) {
            DrawUiText(*font_, lines[i], x + kBubblePadding, line_y, kBubbleText);
        } else {
            // Each line is a piece of the reply, so its first byte's place in
            // the reply is where it points.
            DrawReplyLine(lines[i], static_cast<std::size_t>(lines[i].data() - text.data()), x + kBubblePadding,
                          line_y);
        }
    }
}

// One line of the reply, starting at byte `first` of it, in runs: the
// model's own bytes in the usual colour, the draft model's in teal.
void Dialogue::DrawReplyLine(std::string_view line, std::size_t first, float x, float y) const {
    const std::string& drafted = partner_->drafted;
    if (first + line.size() > drafted.size()) {
        DrawUiText(*font_, line, x, y, kBubbleText);
        return;
    }
    std::size_t run = 0;
    for (std::size_t i = 1; i <= line.size(); ++i) {
        if (i < line.size() && drafted[first + i] == drafted[first + run]) {
            continue;
        }
        DrawUiText(*font_, line.substr(run, i - run), x + TextWidth(*font_, line.substr(0, run)), y,
            drafted[first + run] != 0 ? kDraftedText : kBubbleText);
        run = i;
    }
}

// While typing: the input line with a blinking caret. Otherwise, next to an
// NPC: the player's last message, so the reply has context, and a hint when
// a new message can be sent (or why the last B bought nothing).
void Dialogue::DrawChatBox() const {
    const Rectangle box{ 240, kStreetBottom - 150, 800, 46 };
    if (partner_ != nullptr && !typing_ && !Replying()) {
        std::string hint = "Press E to talk to " + std::string(partner_->persona->name);
        Color color = kHint;
        const std::optional<Offer>& offer = market_->CurrentOffer();
        if (refusal_seconds_ > 0.0f) {
            hint = refusal_;
            color = kRefusal;
        } else if (offer && offer->seller == partner_) {
            hint = std::format("{} for {} credits: B to buy, E to talk", offer->item, offer->price);
        }
        DrawUiText(*font_, hint, box.x + (box.width - TextWidth(*font_, hint)) / 2, box.y + 12, color);
    }
    if (typing_) {
        DrawRectangleRounded(box, 0.25f, 6, kPanel);
        DrawRectangleRoundedLinesEx(box, 0.25f, 6, 2.0f, partner_->color);
        const bool caret_on = static_cast<int>(time_ * 2.0f) % 2 == 0;
        const std::string shown = Tail(*font_, input_, box.width - 60) + (caret_on ? "_" : " ");
        DrawUiText(*font_, "> " + shown, box.x + 16, box.y + 12, RAYWHITE);
    } else if (partner_ != nullptr && Replying() && !player_line_.empty()) {
        const std::string line = Tail(*font_, "You: " + player_line_, box.width - 32);
        DrawRectangleRounded(box, 0.25f, 6, kPanel);
        DrawUiText(*font_, line, box.x + 16, box.y + 12, LIGHTGRAY);
    }
}

}  // namespace murmur::game
