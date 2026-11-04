#include "engine/kv_cache_manager.hpp"

#include <stdexcept>

namespace murmur {

namespace {

int PagesFor(int length, int page_size) {
    return (length + page_size - 1) / page_size;
}

}  // namespace

KvCacheManager::KvCacheManager(int page_count, int page_size)
    : page_size_(page_size), pool_(page_count), prefix_cache_(pool_, page_size) {
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
        throw std::invalid_argument("KvCacheManager: page_size must be a power of two");
    }
}

// Makes `sequence` own enough pages for `length` positions, allocating new
// ones at the end of its page table (and evicting unused cached pages if the
// pool is empty). Returns false when the cache is full; the sequence is then
// as it was, though cached pages evicted along the way stay evicted (they
// were unused, so only the chance to reuse them is lost). The sequence's
// length is the caller's to advance once the positions are actually written.
bool KvCacheManager::Reserve(KvSequence& sequence, int length) {
    if (length < 0) {
        throw std::invalid_argument("KvCacheManager::Reserve: negative length");
    }
    const std::size_t had = sequence.pages.size();
    while (static_cast<int>(sequence.pages.size()) < PagesFor(length, page_size_)) {
        std::optional<PageId> page = pool_.Allocate();
        if (!page && prefix_cache_.EvictOne()) {
            page = pool_.Allocate();
        }
        if (!page) {
            while (sequence.pages.size() > had) {
                pool_.Release(sequence.pages.back());
                sequence.pages.pop_back();
            }
            return false;
        }
        sequence.pages.push_back(*page);
    }
    return true;
}

// Rewinds `sequence` to its first `length` positions, releasing the pages
// past them. Positions from `length` on will be written again, so the page
// holding position `length` must not be shared.
void KvCacheManager::Truncate(KvSequence& sequence, int length) {
    if (length < 0 || length > sequence.length) {
        throw std::invalid_argument("KvCacheManager::Truncate: length out of range");
    }
    const int keep = PagesFor(length, page_size_);
    while (static_cast<int>(sequence.pages.size()) > keep) {
        pool_.Release(sequence.pages.back());
        sequence.pages.pop_back();
    }
    if (length % page_size_ != 0 && pool_.RefCount(sequence.pages.back()) > 1) {
        throw std::logic_error("KvCacheManager::Truncate: would overwrite a shared page");
    }
    sequence.length = length;
}

// Gives every page back (shared pages stay with their other owners).
void KvCacheManager::Release(KvSequence& sequence) {
    for (const PageId page : sequence.pages) {
        pool_.Release(page);
    }
    sequence.pages.clear();
    sequence.length = 0;
}

// Starts an empty sequence on cached pages that already hold a prefix of
// `tokens`, and returns how many positions that covers. At least one token
// is always left to compute, because generation needs the logits of the
// prompt's last token.
int KvCacheManager::AttachPrefix(KvSequence& sequence, std::span<const TokenId> tokens) {
    if (!sequence.pages.empty() || sequence.length != 0) {
        throw std::logic_error("KvCacheManager::AttachPrefix: sequence is not empty");
    }
    if (tokens.empty()) {
        return 0;
    }
    const int max_pages = static_cast<int>(tokens.size() - 1) / page_size_;
    sequence.pages = prefix_cache_.Match(tokens, max_pages);
    sequence.length = static_cast<int>(sequence.pages.size()) * page_size_;
    return sequence.length;
}

// Makes the full pages of `sequence` findable by later prompts that start
// with the same tokens. `tokens` are the sequence's first tokens (at most
// its length); only pages they fill completely are published.
void KvCacheManager::Publish(const KvSequence& sequence, std::span<const TokenId> tokens) {
    if (tokens.size() > static_cast<std::size_t>(sequence.length)) {
        throw std::invalid_argument("KvCacheManager::Publish: more tokens than the sequence holds");
    }
    prefix_cache_.Insert(tokens, sequence.pages);
}

// Empties the prefix cache: pages only it held become free, pages that
// sequences use stay theirs. Benchmarks use it so a run can't reuse an
// earlier run's prompts.
void KvCacheManager::ForgetCachedPrefixes() {
    prefix_cache_.Clear();
}

}  // namespace murmur
