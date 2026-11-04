#include "engine/prefix_cache.hpp"

#include <algorithm>
#include <stdexcept>

namespace murmur {

namespace {

constexpr std::uint64_t kRootKey = 0x9E3779B97F4A7C15ull;

// splitmix64's finalizer: a cheap, well-mixed 64-bit hash step.
std::uint64_t Mix(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}

}  // namespace

PrefixCache::PrefixCache(PagePool& pool, int page_size) : pool_(&pool), page_size_(page_size) {
    if (page_size <= 0) {
        throw std::invalid_argument("PrefixCache: page_size must be positive");
    }
}

PrefixCache::~PrefixCache() {
    Clear();
}

// The cached pages that hold the first full pages of `tokens`, at most
// `max_pages` of them, in order. Each returned page gets one more owner: the
// caller now shares it and must release it like any page it allocated.
std::vector<PageId> PrefixCache::Match(std::span<const TokenId> tokens, int max_pages) {
    std::vector<PageId> pages;
    Key parent = kRootKey;
    const std::size_t full_pages = tokens.size() / page_size_;
    for (std::size_t i = 0; i < full_pages && static_cast<int>(pages.size()) < max_pages; ++i) {
        const std::span<const TokenId> page_tokens = tokens.subspan(i * page_size_, page_size_);
        const Key key = PageKey(parent, page_tokens);
        Entry* entry = Find(key, parent, page_tokens);
        if (entry == nullptr) {
            break;
        }
        pool_->Retain(entry->page);
        Touch(*entry);
        pages.push_back(entry->page);
        parent = key;
    }
    return pages;
}

// Registers the full pages of a sequence: pages[i] holds the keys and values
// of tokens [i * page_size, (i + 1) * page_size). Pages already cached are
// just marked as recently used; when another page already caches the same
// prefix, the existing one is kept and `pages[i]` stays private.
void PrefixCache::Insert(std::span<const TokenId> tokens, std::span<const PageId> pages) {
    Key parent = kRootKey;
    const std::size_t full_pages = std::min(tokens.size() / page_size_, pages.size());
    for (std::size_t i = 0; i < full_pages; ++i) {
        const std::span<const TokenId> page_tokens = tokens.subspan(i * page_size_, page_size_);
        const Key key = PageKey(parent, page_tokens);
        if (Entry* entry = Find(key, parent, page_tokens)) {
            Touch(*entry);
        } else if (entries_.contains(key)) {
            return;   // a 64-bit hash collision: leave the cache as it is
        } else {
            pool_->Retain(pages[i]);
            pages_.insert(pages[i]);
            recency_.push_back(key);
            entries_.emplace(key, Entry{parent, std::vector<TokenId>(page_tokens.begin(), page_tokens.end()),
                                        pages[i], std::prev(recency_.end())});
        }
        parent = key;
    }
}

// Frees the least recently used cached page that no sequence is using (the
// cache holds its only reference). Returns false when every cached page is
// in use.
bool PrefixCache::EvictOne() {
    for (auto it = recency_.begin(); it != recency_.end(); ++it) {
        const auto entry = entries_.find(*it);
        if (pool_->RefCount(entry->second.page) == 1) {
            const PageId page = entry->second.page;
            recency_.erase(it);
            entries_.erase(entry);
            pages_.erase(page);
            pool_->Release(page);
            return true;
        }
    }
    return false;
}

// Drops every cached page (pages still used by sequences stay theirs).
void PrefixCache::Clear() {
    for (const auto& [key, entry] : entries_) {
        pool_->Release(entry.page);
    }
    entries_.clear();
    recency_.clear();
    pages_.clear();
}

PrefixCache::Key PrefixCache::PageKey(Key parent, std::span<const TokenId> page_tokens) const {
    Key key = Mix(parent);
    for (const TokenId token : page_tokens) {
        key = Mix(key ^ static_cast<std::uint32_t>(token));
    }
    return key;
}

PrefixCache::Entry* PrefixCache::Find(Key key, Key parent, std::span<const TokenId> page_tokens) {
    const auto it = entries_.find(key);
    if (it == entries_.end() || it->second.parent != parent || !std::ranges::equal(it->second.tokens, page_tokens)) {
        return nullptr;
    }
    return &it->second;
}

void PrefixCache::Touch(Entry& entry) {
    recency_.splice(recency_.end(), recency_, entry.recency);
}

}  // namespace murmur
