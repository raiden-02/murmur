#pragma once

#include "engine/page_pool.hpp"
#include "model/tokenizer.hpp"

#include <cstdint>
#include <list>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace murmur {

// Remembers which KV pages hold which token prefixes, so a prompt that
// starts like an earlier one (every NPC's system prompt opens with the same
// city lore) reuses those pages instead of recomputing them.
//
// Only full pages are cached: a full page is never written again, so it can
// be shared read-only. A page is identified by its tokens and everything
// before them (a hash chain), so the same words at a different place in a
// prompt never match. The cache owns one reference to every page it holds;
// pages nobody else uses can be evicted, least recently used first.
class PrefixCache {
public:
    PrefixCache(PagePool& pool, int page_size);
    ~PrefixCache();

    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    std::vector<PageId> Match(std::span<const TokenId> tokens, int max_pages);
    void Insert(std::span<const TokenId> tokens, std::span<const PageId> pages);
    bool EvictOne();
    void Clear();

    int Size() const { return static_cast<int>(entries_.size()); }
    bool Contains(PageId page) const { return pages_.contains(page); }

private:
    using Key = std::uint64_t;
    struct Entry {
        Key parent;
        std::vector<TokenId> tokens;   // the page's tokens, to rule out hash collisions
        PageId page;
        std::list<Key>::iterator recency;
    };

    Key PageKey(Key parent, std::span<const TokenId> page_tokens) const;
    Entry* Find(Key key, Key parent, std::span<const TokenId> page_tokens);
    void Touch(Entry& entry);

    PagePool* pool_;
    int page_size_;
    std::unordered_map<Key, Entry> entries_;
    std::list<Key> recency_;   // least recently used at the front
    std::unordered_set<PageId> pages_;
};

}  // namespace murmur
