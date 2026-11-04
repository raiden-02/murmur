#pragma once

#include "engine/page_pool.hpp"
#include "engine/prefix_cache.hpp"
#include "model/tokenizer.hpp"

#include <span>
#include <vector>

namespace murmur {

// One sequence's place in the paged KV cache: its page table (pages[i]
// holds the keys and values of positions [i * page_size, (i + 1) *
// page_size)) and how many positions are filled.
struct KvSequence {
    std::vector<PageId> pages;
    int length = 0;
};

// Host-side bookkeeping for the paged KV cache: which pages each sequence
// owns, which are shared through the prefix cache, and which are free. When
// pages run out it first reclaims cached pages nobody is using.
class KvCacheManager {
public:
    KvCacheManager(int page_count, int page_size);

    bool Reserve(KvSequence& sequence, int length);
    void Truncate(KvSequence& sequence, int length);
    void Release(KvSequence& sequence);
    int AttachPrefix(KvSequence& sequence, std::span<const TokenId> tokens);
    void Publish(const KvSequence& sequence, std::span<const TokenId> tokens);
    void ForgetCachedPrefixes();
    bool EvictCachedPage() { return prefix_cache_.EvictOne(); }

    int PageSize() const { return page_size_; }
    int PageCount() const { return pool_.PageCount(); }
    int FreePages() const { return pool_.FreeCount(); }
    int CachedPages() const { return prefix_cache_.Size(); }
    int RefCount(PageId page) const { return pool_.RefCount(page); }
    bool IsCached(PageId page) const { return prefix_cache_.Contains(page); }
    void CheckInvariants() const { pool_.CheckInvariants(); }

private:
    int page_size_;
    PagePool pool_;
    PrefixCache prefix_cache_;   // after pool_: it releases its pages into the pool when destroyed
};

}  // namespace murmur
