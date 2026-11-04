#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace murmur {

// Index of one fixed-size page of the paged KV cache.
using PageId = std::int32_t;

// Hands out the KV cache's pages and counts how many owners each has. A page
// with no owners is on the free list; sharing a page (a common prompt
// prefix) just adds an owner. Host-side bookkeeping only: the pages' memory
// lives on the GPU (PagedKvCache).
class PagePool {
public:
    explicit PagePool(int page_count);

    std::optional<PageId> Allocate();
    void Retain(PageId page);
    void Release(PageId page);

    int RefCount(PageId page) const;
    int PageCount() const { return static_cast<int>(ref_counts_.size()); }
    int FreeCount() const { return static_cast<int>(free_.size()); }
    void CheckInvariants() const;

private:
    void RequireOwned(PageId page, const char* operation) const;

    std::vector<int> ref_counts_;
    std::vector<PageId> free_;   // a stack: the most recently freed page is reused first
};

}  // namespace murmur
