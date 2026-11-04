#include "model/chat_template.hpp"

#include <string_view>

namespace murmur {

namespace {

// Used by the model's own template when a conversation has no system message.
constexpr std::string_view kDefaultSystemPrompt =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

std::string_view RoleName(ChatRole role) {
    switch (role) {
    case ChatRole::kSystem: return "system";
    case ChatRole::kUser: return "user";
    case ChatRole::kAssistant: return "assistant";
    }
    return "user";
}

void Append(std::vector<TokenId>& out, const std::vector<TokenId>& ids) {
    out.insert(out.end(), ids.begin(), ids.end());
}

}  // namespace

ChatTemplate::ChatTemplate(const Tokenizer& tokenizer)
    : tokenizer_(&tokenizer),
      im_start_(tokenizer.SpecialTokenId("<|im_start|>")),
      im_end_(tokenizer.SpecialTokenId("<|im_end|>")) {}

// A whole conversation. Like the model's own template, a conversation that
// doesn't start with a system message gets the default one.
std::vector<TokenId> ChatTemplate::EncodeConversation(std::span<const ChatMessage> messages,
                                                      bool add_generation_prompt) const {
    std::vector<TokenId> ids;
    if (messages.empty() || messages.front().role != ChatRole::kSystem) {
        Append(ids, EncodeMessage({ChatRole::kSystem, std::string(kDefaultSystemPrompt)}));
    }
    for (const ChatMessage& message : messages) {
        Append(ids, EncodeMessage(message));
    }
    if (add_generation_prompt) {
        Append(ids, EncodeGenerationPrompt());
    }
    return ids;
}

// One complete turn: <|im_start|>{role}\n{content}<|im_end|>\n
std::vector<TokenId> ChatTemplate::EncodeMessage(const ChatMessage& message) const {
    std::vector<TokenId> ids;
    ids.push_back(im_start_);
    // Role header and content form one text segment, exactly as when the
    // rendered template string is tokenized as a whole.
    std::string text(RoleName(message.role));
    text += '\n';
    text += message.content;
    Append(ids, tokenizer_->Encode(text, SpecialTokens::kAsText));
    ids.push_back(im_end_);
    Append(ids, EncodeAfterReply());
    return ids;
}

// <|im_start|>assistant\n : the model continues from here with its reply.
std::vector<TokenId> ChatTemplate::EncodeGenerationPrompt() const {
    std::vector<TokenId> ids;
    ids.push_back(im_start_);
    Append(ids, tokenizer_->Encode("assistant\n", SpecialTokens::kAsText));
    return ids;
}

// The newline that follows <|im_end|>. When the model generates a reply, it
// produces <|im_end|> itself; this closes the turn so the next one can start.
std::vector<TokenId> ChatTemplate::EncodeAfterReply() const {
    return tokenizer_->Encode("\n", SpecialTokens::kAsText);
}

}  // namespace murmur
