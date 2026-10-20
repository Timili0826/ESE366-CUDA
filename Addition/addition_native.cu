// ESE366: Matrix addition, FP32 baseline.
// Build: nvcc -O3 -lineinfo addition_native.cu -o addition_native
// Run: ./addition_native [N=1000] [threads_per_block=256]
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(call) do {                                             \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        std::fprintf(stderr, "Line %d: %s\n", __LINE__,                \
                     cudaGetErrorString(err));                       \
        std::exit(EXIT_FAILURE);                                    \
    }                                                               \
} while (0)

__global__ void additionNative(const float* A, const float* B,
                               float* C, size_t count)
{
    // [A: mapping] Flatten an N-by-N row-major matrix into N*N elements.
    // One thread owns one output. No row/column division is necessary.
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // [B: boundary] Required when count is not divisible by blockDim.x.
    // With blocks aligned to whole warps, only the final warp can be partial.
    if (i < count) {
        // [C: memory] Adjacent threads access adjacent FP32 values in all arrays.
        // Two reads + one write = 12 useful bytes per output element.
        // No cross-thread input reuse: shared-memory staging adds work here.
        C[i] = A[i] + B[i];
    }
    // [D: synchronization] Independent outputs require no __syncthreads().
    // [E: work per thread] There is no loop to unroll in this baseline.
    // Later: test several elements/thread, preserving coalesced accesses.
    // More work/thread can reduce blocks but may reduce available parallelism.
    // [F: precision] A separate FP64 version can test precision/time trade-offs.
    // Keep precision fixed when comparing launch or work-distribution changes.
}

static int positiveInt(const char* arg)
{
    std::string s(arg);
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Use positive integer arguments.");
    int value = std::stoi(s);
    if (value <= 0) throw std::invalid_argument("Arguments must be positive.");
    return value;
}

int main(int argc, char** argv)
{
    try {
        if (argc > 3)
            throw std::invalid_argument("Usage: ./addition_native [N] [threads]");
        int N = argc > 1 ? positiveInt(argv[1]) : 1000;
        int threads = argc > 2 ? positiveInt(argv[2]) : 256;
        size_t n = static_cast<size_t>(N);
        if (n > std::numeric_limits<size_t>::max() / n / sizeof(float))
            throw std::invalid_argument("Matrix byte size overflows size_t.");
        size_t count = n * n;
        size_t bytes = count * sizeof(float);

        cudaDeviceProp prop{};
        CHECK(cudaGetDeviceProperties(&prop, 0));
        if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0])
            throw std::invalid_argument("Block size exceeds device limits.");

        // [G: launch tuning] Test 128, 256 and 512 threads with the same input.
        // Grid size is derived from count and threads, rather than held fixed.
        size_t blocks = count / threads + (count % threads != 0);
        if (blocks > static_cast<size_t>(prop.maxGridSize[0]))
            throw std::invalid_argument("Grid size exceeds device limits.");

        std::vector<float> hA(count), hB(count), hC(count);
        for (size_t i = 0; i < count; ++i) {
            // Signed fractional data exercises FP32 rounding and cancellation.
            hA[i] = (static_cast<int>(i % 2001) - 1000) / 1000.0f;
            hB[i] = (static_cast<int>(((i % 1999) * 17) % 1999) - 999) / 1000.0f;
        }
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), bytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), bytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dC), bytes));
        CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dC, 0xff, bytes));

        const int warmups = 10;
        const int repeats = 100;
        for (int i = 0; i < warmups; ++i)
            additionNative<<<static_cast<unsigned>(blocks), threads>>>(dA, dB, dC, count);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // [H: timing] CUDA events cover 100 launches, divided by 100.
        // Allocation, initialization, copies and validation are excluded.
        // No device-wide synchronization is inserted between timed launches.
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i)
            additionNative<<<static_cast<unsigned>(blocks), threads>>>(dA, dB, dC, count);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        double averageMs = static_cast<double>(elapsed) / repeats;

        CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        // [I: correctness] FP64 reference sums the actual stored FP32 inputs.
        // Check every element using abs_error <= atol + rtol * abs(reference).
        // Normalized infinity error = max absolute error / max abs reference;
        // this is not the maximum of the individual relative errors.
        const double atol = 1e-6, rtol = 1e-5;
        double maxAbs = 0.0, maxRef = 0.0;
        size_t mismatches = 0;
        for (size_t i = 0; i < count; ++i) {
            double ref = static_cast<double>(hA[i]) + static_cast<double>(hB[i]);
            double error = std::abs(static_cast<double>(hC[i]) - ref);
            maxRef = std::max(maxRef, std::abs(ref));
            if (!std::isfinite(hC[i])) {
                maxAbs = std::numeric_limits<double>::infinity();
                ++mismatches;
            } else {
                maxAbs = std::max(maxAbs, error);
                if (error > atol + rtol * std::abs(ref)) ++mismatches;
            }
        }
        double normError = maxRef > 0.0 ? maxAbs / maxRef : maxAbs;
        // [J: bandwidth] Useful bytes/time, not a measured DRAM bandwidth.
        // Cache effects can influence repeated launches on the same arrays.
        double effectiveGBs = 3.0 * static_cast<double>(bytes) / (averageMs * 1e6);
        size_t warpsPerBlock = (static_cast<size_t>(threads) + 31) / 32;
        std::printf("GPU: %s\n", prop.name);
        std::printf("N=%d, threads/block=%d, grid blocks=%zu\n", N, threads, blocks);
        std::printf("Launched threads=%zu, warps/block=%zu, launched warps=%zu\n",
                    blocks * threads, warpsPerBlock, blocks * warpsPerBlock);
        std::printf("Valid elements=%zu; explicit shared memory/block=0 bytes\n", count);
        std::printf("Warm-up kernel launches: %d\n", warmups);
        std::printf("Average kernel time: %.6f ms (%d repetitions)\n", averageMs, repeats);
        std::printf("Effective bandwidth: %.3f GB/s (useful bytes/time)\n", effectiveGBs);
        std::printf("Max absolute error: %.9g; normalized infinity error: %.9g\n", maxAbs, normError);
        std::printf("Tolerance per element: %.1e + %.1e * abs(reference)\n", atol, rtol);
        std::printf("Check: %s, mismatches=%zu\n", mismatches == 0 ? "PASS" : "FAIL", mismatches);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        CHECK(cudaFree(dA));
        CHECK(cudaFree(dB));
        CHECK(cudaFree(dC));
        return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return EXIT_FAILURE;
    }
}
