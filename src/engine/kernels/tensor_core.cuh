#pragma once

// The handful of PTX instructions the tensor-core kernels share: shared
// memory addresses, asynchronous copies, ldmatrix and the FP16 mma.

#include <cuda_fp16.h>

namespace murmur {

__device__ __forceinline__ unsigned SharedAddress(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

// Asynchronous global-to-shared copies (they bypass registers and let the
// next tiles load while this one computes). With valid == false the 16
// bytes are zero-filled and nothing is read.
__device__ __forceinline__ void CopyAsync16(void* shared, const void* global, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(SharedAddress(shared)), "l"(global),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void CopyAsync4(void* shared, const void* global) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(SharedAddress(shared)), "l"(global));
}

__device__ __forceinline__ void CommitCopies() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Waits until at most `kPending` committed groups are still in flight.
template <int kPending>
__device__ __forceinline__ void WaitCopies() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

// Four 8x8 FP16 matrices from shared memory, one row address per lane
// (lanes 8i .. 8i + 7 give the rows of matrix i): register i holds matrix i
// in the mma fragment layout (lane l: row l / 4, columns 2 (l % 4), +1).
__device__ __forceinline__ void LoadMatrixX4(unsigned (&r)[4], const void* shared) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(SharedAddress(shared)));
}

// The same, each matrix transposed (lane l: column l / 4, rows 2 (l % 4),
// +1): for a B operand stored with its n dimension contiguous.
__device__ __forceinline__ void LoadMatrixX4Transposed(unsigned (&r)[4], const void* shared) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(SharedAddress(shared)));
}

// c += a . b on tensor cores: a 16x16 FP16, b 16x8 FP16, c 16x8 FP32.
__device__ __forceinline__ void Mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned Bits(__half2 h) {
    return *reinterpret_cast<const unsigned*>(&h);
}

}  // namespace murmur
