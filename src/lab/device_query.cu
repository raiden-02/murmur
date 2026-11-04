// Toolchain check: prints the properties of each CUDA device and the
// theoretical memory bandwidth that later phases measure against.

#include "lab/lab.hpp"

#include "common/cuda_check.hpp"
#include "common/device_info.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <optional>

namespace {

int Attribute(cudaDeviceAttr attribute, int device) {
	int value = 0;
	murmur::CudaCheck(cudaDeviceGetAttribute(&value, attribute, device));
	return value;
}

}  // namespace

namespace murmur::lab {

// Prints every device's properties. CUDA failures throw (see lab_main.cpp).
int RunDeviceQuery(Args args) {
	if (const std::optional<int> exit_code = ArgParser("murmur-lab devices").Parse(args)) {
		return *exit_code;
	}
	int device_count = 0;
	murmur::CudaCheck(cudaGetDeviceCount(&device_count));

	int runtime_version = 0;
	int driver_version = 0;
	murmur::CudaCheck(cudaRuntimeGetVersion(&runtime_version));
	murmur::CudaCheck(cudaDriverGetVersion(&driver_version));
	std::printf("CUDA runtime %d.%d, driver supports %d.%d, %d device(s)\n\n",
		runtime_version / 1000, (runtime_version % 1000) / 10,
		driver_version / 1000, (driver_version % 1000) / 10,
		device_count);

	for (int device = 0; device < device_count; ++device) {
		cudaDeviceProp properties{};
		murmur::CudaCheck(cudaGetDeviceProperties(&properties, device));

		const int bus_width_bits = Attribute(cudaDevAttrGlobalMemoryBusWidth, device);
		const double bandwidth_gb_per_s = murmur::PeakMemoryBandwidth(device) / 1e9;

		std::printf("Device %d: %s\n", device, properties.name);
		std::printf("  Compute capability   %d.%d\n", properties.major, properties.minor);
		std::printf("  Streaming multiprocs %d\n", properties.multiProcessorCount);
		std::printf("  Max threads per SM   %d (at the same time)\n",
			properties.maxThreadsPerMultiProcessor);
		std::printf("  Max blocks per SM    %d (at the same time)\n",
			Attribute(cudaDevAttrMaxBlocksPerMultiprocessor, device));
		std::printf("  Max threads per block %d\n", properties.maxThreadsPerBlock);
		std::printf("  Global memory        %.1f GiB\n",
			properties.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
		std::printf("  L2 cache             %.1f MiB\n",
			properties.l2CacheSize / (1024.0 * 1024.0));
		std::printf("  Shared mem per block %zu KiB (opt-in max %zu KiB)\n",
			properties.sharedMemPerBlock / 1024,
			properties.sharedMemPerBlockOptin / 1024);
		std::printf("  Memory bus           %d-bit\n", bus_width_bits);
		std::printf("  Peak bandwidth       %.0f GB/s (theoretical)\n\n", bandwidth_gb_per_s);
	}

	return EXIT_SUCCESS;
}

}  // namespace murmur::lab
