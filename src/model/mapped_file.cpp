#include "model/mapped_file.hpp"

#ifndef _WIN32
#error "MappedFile is implemented for Windows only"
#endif

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace murmur {

namespace {

// Closes a Win32 handle on scope exit. The mapping view stays valid after
// both the file and the mapping handles are closed.
class HandleCloser {
public:
    explicit HandleCloser(HANDLE handle) : handle_(handle) {}
    ~HandleCloser() {
        if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
    }
    HandleCloser(const HandleCloser&) = delete;
    HandleCloser& operator=(const HandleCloser&) = delete;

private:
    HANDLE handle_;
};

[[noreturn]] void ThrowLastError(const std::string& what, const std::filesystem::path& path) {
    throw std::runtime_error(what + " failed for " + path.string() + " (Win32 error " +
                             std::to_string(GetLastError()) + ")");
}

}  // namespace

// Maps the whole file read-only. Throws std::runtime_error if the file can't
// be opened or mapped. An empty file maps to an empty span (Windows can't
// map zero bytes).
MappedFile::MappedFile(const std::filesystem::path& path) {
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        ThrowLastError("CreateFileW", path);
    }
    const HandleCloser file_closer(file);

    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size)) {
        ThrowLastError("GetFileSizeEx", path);
    }
    if (size.QuadPart == 0) {
        return;
    }

    const HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        ThrowLastError("CreateFileMappingW", path);
    }
    const HandleCloser mapping_closer(mapping);

    const void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        ThrowLastError("MapViewOfFile", path);
    }
    data_ = static_cast<const std::byte*>(view);
    size_ = static_cast<std::size_t>(size.QuadPart);
}

MappedFile::~MappedFile() {
    Release();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        Release();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

// The file's contents. Valid for as long as this object (or the object it is
// moved into) lives.
std::span<const std::byte> MappedFile::Bytes() const {
    return {data_, size_};
}

void MappedFile::Release() noexcept {
    if (data_ != nullptr) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
        size_ = 0;
    }
}

}  // namespace murmur
