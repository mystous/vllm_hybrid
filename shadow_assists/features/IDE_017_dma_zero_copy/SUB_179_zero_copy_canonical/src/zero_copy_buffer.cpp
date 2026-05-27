// IDE_017 / TSK_029 — Zero-Copy CPU compute path
// cudaHostAllocMapped dual-access pinned buffer (CPU + GPU 동시 read/write).
//
// 본 SUB_179 의 lever:
//   - small data (4 KB ~ 256 KB) 에서 cudaMemcpyAsync round-trip (fixed 35 μs overhead)
//     vs zero-copy GPU 의 직접 read latency 비교
//   - SUB_166 의 1 MB crossover 분석 재검증 (DMA 가 zero-copy 보다 빨라지는 점)
//
// build:
//   nvcc -arch=sm_90 -O3 -std=c++17 -Xcompiler -fPIC -shared \
//        -o libzero_copy.so zero_copy_buffer.cpp -lcudart
//
// status: SUB_179 ⏳ 구현 + microbench (1-run)

#include <cuda_runtime.h>
#include <time.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace vllm_hybrid_zc {

// ──────────────────────────────────────────────────────────────────────
// ZeroCopyBuffer — cudaHostAllocMapped dual-access region
//   - host_ptr: CPU 가 직접 read/write (page-locked, mapped to PCIe BAR)
//   - dev_ptr:  GPU 가 cudaHostGetDevicePointer 로 같은 물리 메모리를 가리킴
//   - coherence: weakly-coherent; PCIe BAR access → CPU↔GPU 가 동기화 필요
// ──────────────────────────────────────────────────────────────────────

struct ZeroCopyBuffer {
    void*  host_ptr;   // CPU-side pointer (page-locked)
    void*  dev_ptr;    // GPU-side device pointer (mapped)
    size_t size;
};

// alloc — returns 0 on success
extern "C" int zc_alloc(ZeroCopyBuffer* zcb, size_t size) {
    if (!zcb) return -1;
    void* host = nullptr;
    cudaError_t rc = cudaHostAlloc(&host, size, cudaHostAllocMapped);
    if (rc != cudaSuccess || host == nullptr) {
        std::fprintf(stderr, "[zc_alloc] cudaHostAlloc failed: %s\n",
                     cudaGetErrorString(rc));
        return -2;
    }
    void* dev = nullptr;
    rc = cudaHostGetDevicePointer(&dev, host, 0);
    if (rc != cudaSuccess || dev == nullptr) {
        cudaFreeHost(host);
        std::fprintf(stderr, "[zc_alloc] cudaHostGetDevicePointer failed: %s\n",
                     cudaGetErrorString(rc));
        return -3;
    }
    zcb->host_ptr = host;
    zcb->dev_ptr  = dev;
    zcb->size     = size;
    return 0;
}

extern "C" int zc_free(ZeroCopyBuffer* zcb) {
    if (!zcb || !zcb->host_ptr) return -1;
    cudaError_t rc = cudaFreeHost(zcb->host_ptr);
    zcb->host_ptr = nullptr;
    zcb->dev_ptr  = nullptr;
    zcb->size     = 0;
    return (rc == cudaSuccess) ? 0 : -2;
}

// Probe device for mapped-host support — required for cudaHostAllocMapped to function.
extern "C" int zc_check_device_support(int dev_id) {
    int can_map = 0;
    cudaError_t rc = cudaDeviceGetAttribute(&can_map,
                                            cudaDevAttrCanMapHostMemory,
                                            dev_id);
    if (rc != cudaSuccess) return -1;
    return can_map;  // 1 = supported, 0 = not supported
}

// ──────────────────────────────────────────────────────────────────────
// Trivial GPU kernel that reads `n` u32 from src and writes them to dst.
// Used by microbench to measure GPU-side access latency to:
//   (a) zero-copy host-mapped memory (dev_ptr from cudaHostGetDevicePointer)
//   (b) device-resident memory (cudaMalloc)
// ──────────────────────────────────────────────────────────────────────

__global__ void copy_u32_kernel(const uint32_t* __restrict__ src,
                                uint32_t* __restrict__ dst,
                                int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        dst[i] = src[i] + 1u;  // add 1 to force materialization (no DCE)
    }
}

// Sum reduction (write to single u32 sink) — read-heavy probe.
__global__ void sum_u32_kernel(const uint32_t* __restrict__ src,
                               uint32_t* __restrict__ sink,
                               int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    uint32_t acc = 0;
    for (int i = tid; i < n; i += stride) acc += src[i];
    atomicAdd(sink, acc);
}

// ──────────────────────────────────────────────────────────────────────
// Microbench primitive
// Mode A — zero-copy GPU read latency:
//   1) CPU fills host buffer (RAM-side; mapped region is BAR-mapped pinned RAM)
//   2) GPU launches kernel that reads via dev_ptr (PCIe BAR)
//   3) record time from CPU write completion → GPU kernel completion
//
// Mode B — DMA push + GPU read latency (baseline):
//   1) CPU fills pageable / pinned source buffer
//   2) cudaMemcpyAsync host→device
//   3) GPU launches kernel that reads from device memory
//   4) record time from CPU write completion → GPU kernel completion
//
// Both modes pay the kernel launch fixed cost; difference = transfer model.
// ──────────────────────────────────────────────────────────────────────

extern "C" double zc_bench_mode_zerocopy(size_t bytes, int iters, int gpu_id) {
    if (cudaSetDevice(gpu_id) != cudaSuccess) return -1.0;
    if (bytes < sizeof(uint32_t)) return -2.0;
    int n = (int)(bytes / sizeof(uint32_t));

    // zero-copy source (host-mapped)
    void* src_host = nullptr;
    cudaError_t rc = cudaHostAlloc(&src_host, bytes, cudaHostAllocMapped);
    if (rc != cudaSuccess) return -3.0;
    void* src_dev = nullptr;
    rc = cudaHostGetDevicePointer(&src_dev, src_host, 0);
    if (rc != cudaSuccess) { cudaFreeHost(src_host); return -4.0; }

    // device-side dst for kernel
    void* dst_dev = nullptr;
    rc = cudaMalloc(&dst_dev, bytes);
    if (rc != cudaSuccess) { cudaFreeHost(src_host); return -5.0; }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // warmup
    for (int w = 0; w < 5; ++w) {
        // CPU update
        std::memset(src_host, (w & 0xFF), bytes);
        // GPU read via zero-copy device pointer
        copy_u32_kernel<<<32, 256, 0, stream>>>(
            (uint32_t*)src_dev, (uint32_t*)dst_dev, n);
    }
    cudaStreamSynchronize(stream);

    double total_ms = 0.0;
    for (int it = 0; it < iters; ++it) {
        // freshly stamp the CPU side (force CPU write fence)
        std::memset(src_host, (it & 0xFF), bytes);
        // Make sure CPU writes are visible to GPU (PCIe ordered store; on x86
        // sfence may be needed for store-buffer drain).
        __sync_synchronize();

        cudaEventRecord(e0, stream);
        copy_u32_kernel<<<32, 256, 0, stream>>>(
            (uint32_t*)src_dev, (uint32_t*)dst_dev, n);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);

        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        total_ms += ms;
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(stream);
    cudaFree(dst_dev);
    cudaFreeHost(src_host);

    // return median? we return mean; caller measures wall-time median externally
    return total_ms / iters;
}

extern "C" double zc_bench_mode_dma(size_t bytes, int iters, int gpu_id) {
    if (cudaSetDevice(gpu_id) != cudaSuccess) return -1.0;
    if (bytes < sizeof(uint32_t)) return -2.0;
    int n = (int)(bytes / sizeof(uint32_t));

    // pinned host source (regular page-locked, no mapping)
    void* src_host = nullptr;
    if (cudaHostAlloc(&src_host, bytes, cudaHostAllocDefault) != cudaSuccess) return -3.0;
    // device-side mirror
    void* src_dev = nullptr;
    if (cudaMalloc(&src_dev, bytes) != cudaSuccess) {
        cudaFreeHost(src_host); return -4.0;
    }
    // device-side dst for kernel
    void* dst_dev = nullptr;
    if (cudaMalloc(&dst_dev, bytes) != cudaSuccess) {
        cudaFree(src_dev); cudaFreeHost(src_host); return -5.0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // warmup
    for (int w = 0; w < 5; ++w) {
        std::memset(src_host, (w & 0xFF), bytes);
        cudaMemcpyAsync(src_dev, src_host, bytes, cudaMemcpyHostToDevice, stream);
        copy_u32_kernel<<<32, 256, 0, stream>>>(
            (uint32_t*)src_dev, (uint32_t*)dst_dev, n);
    }
    cudaStreamSynchronize(stream);

    double total_ms = 0.0;
    for (int it = 0; it < iters; ++it) {
        std::memset(src_host, (it & 0xFF), bytes);
        __sync_synchronize();

        cudaEventRecord(e0, stream);
        cudaMemcpyAsync(src_dev, src_host, bytes, cudaMemcpyHostToDevice, stream);
        copy_u32_kernel<<<32, 256, 0, stream>>>(
            (uint32_t*)src_dev, (uint32_t*)dst_dev, n);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);

        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        total_ms += ms;
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(stream);
    cudaFree(dst_dev);
    cudaFree(src_dev);
    cudaFreeHost(src_host);

    return total_ms / iters;
}

// Variant: device-only baseline (no host transfer). Lower bound for kernel cost.
extern "C" double zc_bench_mode_devonly(size_t bytes, int iters, int gpu_id) {
    if (cudaSetDevice(gpu_id) != cudaSuccess) return -1.0;
    if (bytes < sizeof(uint32_t)) return -2.0;
    int n = (int)(bytes / sizeof(uint32_t));

    void* src_dev = nullptr;
    if (cudaMalloc(&src_dev, bytes) != cudaSuccess) return -3.0;
    void* dst_dev = nullptr;
    if (cudaMalloc(&dst_dev, bytes) != cudaSuccess) {
        cudaFree(src_dev); return -4.0;
    }
    cudaMemset(src_dev, 0x5A, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    for (int w = 0; w < 5; ++w) {
        copy_u32_kernel<<<32, 256, 0, stream>>>(
            (uint32_t*)src_dev, (uint32_t*)dst_dev, n);
    }
    cudaStreamSynchronize(stream);

    double total_ms = 0.0;
    for (int it = 0; it < iters; ++it) {
        cudaEventRecord(e0, stream);
        copy_u32_kernel<<<32, 256, 0, stream>>>(
            (uint32_t*)src_dev, (uint32_t*)dst_dev, n);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        total_ms += ms;
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(stream);
    cudaFree(dst_dev);
    cudaFree(src_dev);
    return total_ms / iters;
}

// CPU-side direct write to host-mapped buffer; measures CPU-side latency only.
extern "C" double zc_bench_cpu_write(size_t bytes, int iters) {
    void* host = nullptr;
    if (cudaHostAlloc(&host, bytes, cudaHostAllocMapped) != cudaSuccess) return -1.0;

    auto now_ns = []() -> uint64_t {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
    };

    // warmup
    for (int w = 0; w < 5; ++w) std::memset(host, w & 0xFF, bytes);

    uint64_t t0 = now_ns();
    for (int i = 0; i < iters; ++i) {
        std::memset(host, i & 0xFF, bytes);
    }
    uint64_t t1 = now_ns();

    cudaFreeHost(host);
    return ((double)(t1 - t0) / 1e6) / iters;  // ms per iter
}

}  // namespace vllm_hybrid_zc
