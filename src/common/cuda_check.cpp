#include "cuda_check.hpp"

#include <stdexcept>
#include <string>

namespace murmur {
    // Throws std::runtime_error if error is not cudaSuccess.
    // Message: "<file>:<line> <function>: <error name>: <error string>".
    // `where` defaults (in the header) to the caller's location.
    void CudaCheck(cudaError_t error, std::source_location where) {
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string(where.file_name()) + ":" +
                std::to_string(where.line()) + " " +
                where.function_name() + ": " +
                cudaGetErrorName(error) + ": " +
                cudaGetErrorString(error));
        }
    }

    // Call right after a kernel launch. Catches launch errors only (bad
    // configuration, e.g. too many threads per block). Errors while the
    // kernel runs appear later, at the next synchronizing call.
    void CheckKernelLaunch(std::source_location where) {
        cudaError_t error = cudaGetLastError();
        CudaCheck(error, where);
    }
}
