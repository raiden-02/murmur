#pragma once

#include <cuda_runtime.h>

#include <functional>

namespace murmur {

// A recorded sequence of GPU work that can be replayed with one launch.
// Owns the graph and its executable instance. Move-only.
class CudaGraph {
public:
    static CudaGraph Capture(cudaStream_t stream, const std::function<void()>& enqueue);

    CudaGraph() = default;
    ~CudaGraph();

    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    CudaGraph(CudaGraph&& other) noexcept;
    CudaGraph& operator=(CudaGraph&& other) noexcept;

    bool Empty() const { return exec_ == nullptr; }
    void Launch(cudaStream_t stream) const;

private:
    void Release() noexcept;

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
};

}  // namespace murmur
