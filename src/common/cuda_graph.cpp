#include "common/cuda_graph.hpp"

#include "common/cuda_check.hpp"

#include <stdexcept>
#include <utility>

namespace murmur {

// Records everything `enqueue` queues on `stream` (kernels, async copies)
// into a graph and instantiates it, without running any of it. Kernel
// arguments and buffer addresses are frozen at capture time, so anything
// that must change between replays has to be read from memory the graph
// points at. If `enqueue` throws, capture is ended and the stream is usable
// again before the exception propagates.
CudaGraph CudaGraph::Capture(cudaStream_t stream, const std::function<void()>& enqueue) {
    CudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CudaGraph result;
    try {
        enqueue();
    } catch (...) {
        cudaGraph_t partial = nullptr;
        static_cast<void>(cudaStreamEndCapture(stream, &partial));
        static_cast<void>(cudaGraphDestroy(partial));
        static_cast<void>(cudaGetLastError());   // clear the capture's error state
        throw;
    }
    CudaCheck(cudaStreamEndCapture(stream, &result.graph_));
    CudaCheck(cudaGraphInstantiate(&result.exec_, result.graph_, 0));
    return result;
}

CudaGraph::~CudaGraph() {
    Release();
}

CudaGraph::CudaGraph(CudaGraph&& other) noexcept
    : graph_(std::exchange(other.graph_, nullptr)), exec_(std::exchange(other.exec_, nullptr)) {}

CudaGraph& CudaGraph::operator=(CudaGraph&& other) noexcept {
    if (this != &other) {
        Release();
        graph_ = std::exchange(other.graph_, nullptr);
        exec_ = std::exchange(other.exec_, nullptr);
    }
    return *this;
}

// Queues the whole recorded sequence on `stream` as one launch.
void CudaGraph::Launch(cudaStream_t stream) const {
    if (exec_ == nullptr) {
        throw std::logic_error("CudaGraph::Launch: empty graph");
    }
    CudaCheck(cudaGraphLaunch(exec_, stream));
}

void CudaGraph::Release() noexcept {
    if (exec_ != nullptr) {
        static_cast<void>(cudaGraphExecDestroy(exec_));
        exec_ = nullptr;
    }
    if (graph_ != nullptr) {
        static_cast<void>(cudaGraphDestroy(graph_));
        graph_ = nullptr;
    }
}

}  // namespace murmur
