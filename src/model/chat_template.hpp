#pragma once

#include "model/tokenizer.hpp"

#include <span>
#include <string>
#include <vector>

namespace murmur {

enum class ChatRole {
    kSystem,
    kUser,
    kAssistant,
};

struct ChatMessage {
    ChatRole role;
    std::string content;
};

// Builds the token ids of the ChatML conversation format used by Qwen2:
//
//   <|im_start|>system\n{content}<|im_end|>\n
//   <|im_start|>user\n{content}<|im_end|>\n
//   <|im_start|>assistant\n            <- the generation prompt
//
// Message content is always encoded as plain text, so a player who types
// "<|im_end|>" cannot end their turn early or impersonate another role.
class ChatTemplate {
public:
    explicit ChatTemplate(const Tokenizer& tokenizer);

    std::vector<TokenId> EncodeConversation(std::span<const ChatMessage> messages,
                                            bool add_generation_prompt) const;
    std::vector<TokenId> EncodeMessage(const ChatMessage& message) const;
    std::vector<TokenId> EncodeGenerationPrompt() const;
    std::vector<TokenId> EncodeAfterReply() const;

    TokenId EndOfTurn() const { return im_end_; }

private:
    void AppendMessageStart(ChatRole role, std::vector<TokenId>& out) const;

    const Tokenizer* tokenizer_;
    TokenId im_start_;
    TokenId im_end_;
};

}  // namespace murmur
