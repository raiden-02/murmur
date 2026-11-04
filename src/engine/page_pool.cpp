#include "engine/page_pool.hpp"

#include <stdexcept>
#include <string>

namespace murmur {

// All pages start free. They are handed out lowest id first, which keeps a
// fresh cache's pages together (and its X-ray map readable).
PagePool::PagePool(int page_count) {
    if (page_count <= 0) {
        throw std::invalid_argument("PagePool: page_count must be positive");
    }
    ref_counts_.assign(page_count, 0);
    free_.reserve(page_count);
    for (PageId page = page_count - 1; page >= 0; --page) {
        free_.push_back(page);
    }
}

// A free page with one owner, or nullopt when every page is in use.
std::optional<PageId> PagePool::Allocate() {
    if (free_.empty()) {
        return std::nullopt;
    }
    const PageId page = free_.back();
    free_.pop_back();
    ref_counts_[page] = 1;
    return page;
}

// Adds an owner to a page that already has one (sharing it).
void PagePool::Retain(PageId page) {
    RequireOwned(page, "Retain");
    ++ref_counts_[page];
}

// Removes an owner; the last one returns the page to the free list.
void PagePool::Release(PageId page) {
    RequireOwned(page, "Release");
    if (--ref_counts_[page] == 0) {
        free_.push_back(page);
    }
}

int PagePool::RefCount(PageId page) const {
    if (page < 0 || page >= PageCount()) {
        throw std::out_of_range("PagePool: page out of range");
    }
    return ref_counts_[page];
}

// Throws std::logic_error unless every page is either owned or on the free
// list exactly once. Cheap enough (one pass) for tests and soak runs.
void PagePool::CheckInvariants() const {
    std::vector<int> listed(ref_counts_.size(), 0);
    for (const PageId page : free_) {
        if (page < 0 || page >= PageCount() || ++listed[page] > 1) {
            throw std::logic_error("PagePool: free list holds an invalid or repeated page");
        }
    }
    for (PageId page = 0; page < PageCount(); ++page) {
        const bool free = listed[page] == 1;
        if (ref_counts_[page] < 0 || free == (ref_counts_[page] > 0)) {
            throw std::logic_error("PagePool: page " + std::to_string(page) + " is both free and owned, or neither");
        }
    }
}

// Retaining or releasing a page nobody owns is a bookkeeping bug (a double
// free, or use after free): fail loudly instead of corrupting the free list.
void PagePool::RequireOwned(PageId page, const char* operation) const {
    if (page < 0 || page >= PageCount()) {
        throw std::out_of_range(std::string("PagePool::") + operation + ": page out of range");
    }
    if (ref_counts_[page] <= 0) {
        throw std::logic_error(std::string("PagePool::") + operation + ": page " + std::to_string(page) + " is free");
    }
}

}  // namespace murmur
