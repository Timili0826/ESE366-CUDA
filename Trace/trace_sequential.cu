// ESE 366: matrix trace with sequential-addressing reduction (FP32).
// Build: nvcc -O3 -lineinfo trace_sequential.cu -o trace_sequential
// Run: ./trace_sequential [N=4096] [threads_per_block=256]
// Conceptual reference (parallel reduction, independently written code):
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

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

// First launch: input = full matrix, count = N, stride = N+1.
// Later launches: input = partial sums, stride = 1.
__global__ void traceReduceSequential(const float* input, float* partial,
                                  size_t count, size_t stride)
{
    extern __shared__ float values[];
    unsigned tid = threadIdx.x;
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

    // [A: GLOBAL MEMORY] Diagonal elements have a stride of N+1 floats.
    // Neighbouring threads do not read neighbouring matrix addresses.
    // Packing the diagonal adds work; evaluate its total cost before adopting it.
    // [B: BOUNDARY] Out-of-range threads contribute zero. Do not return early:
    // all block threads must still reach the barriers below.
    values[tid] = (i < count) ? input[i * stride] : 0.0f;
    __syncthreads();

    // [C: SEQUENTIAL ADDRESSING] Active threads are contiguous:
    // 0..127, 0..63, 0..31, ... for a 256-thread block.
    // This removes the modulo condition and reduces mixed active/inactive warps.
    // The final rounds still have partial-warp participation; divergence is not zero.
    // [D: UNROLLING] No manual unrolling in this experiment.
    // [E: SYNCHRONIZATION] Keep the same number and placement of barriers
    // as the baseline to isolate the reduction-layout change.
    for (unsigned s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s)
            values[tid] += values[tid + s];
        __syncthreads();
    }

    // One partial sum per block, no atomic addition needed.
    if (tid == 0)
        partial[blockIdx.x] = values[0];
}

static int positiveInt(const char* arg)
{
    std::string s(arg);
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Use positive integer arguments.");
    int value = std::stoi(s);
    if (value <= 0)
        throw std::invalid_argument("Arguments must be positive.");
    return value;
}

static size_t ceilDiv(size_t count, size_t block)
{
    return count / block + (count % block != 0);
}

int main(int argc, char** argv)
{
    try {
        if (argc > 3)
            throw std::invalid_argument("Usage: ./trace_sequential [N] [threads]");
        int N = argc > 1 ? positiveInt(argv[1]) : 4096;
        int threads = argc > 2 ? positiveInt(argv[2]) : 256;

        // [F: BLOCK SIZE] This implementation requires a power-of-two block size >=2.
        // Compare 128, 256 and 512 while keeping N and inputs fixed.
        if (threads < 2 || (threads & (threads - 1)) != 0)
            throw std::invalid_argument("Threads must be a power of two, at least 2.");
        if (static_cast<size_t>(N) > std::numeric_limits<size_t>::max() /
            static_cast<size_t>(N) / sizeof(float))
            throw std::invalid_argument("Matrix byte size overflows size_t.");

        size_t count = static_cast<size_t>(N) * N;
        size_t bytes = count * sizeof(float);
        size_t firstBlocks = ceilDiv(static_cast<size_t>(N), threads);
        size_t sharedBytes = static_cast<size_t>(threads) * sizeof(float);

        cudaDeviceProp prop{};
        CHECK(cudaGetDeviceProperties(&prop, 0));
        if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0])
            throw std::invalid_argument("Block size exceeds device limits.");
        if (firstBlocks > static_cast<size_t>(prop.maxGridSize[0]))
            throw std::invalid_argument("Grid size exceeds device limits.");

        // The input is a full N*N matrix, not a pre-extracted diagonal vector.
        std::vector<float> hMatrix(count);
        for (size_t i = 0; i < count; ++i)
            hMatrix[i] = (static_cast<int>(i % 2001) - 1000) / 1000.0f;
        for (int i = 0; i < N; ++i)
            hMatrix[static_cast<size_t>(i) * N + i] = (1 + i % 997) / 1000.0f;

        // [G: PRECISION] FP64 reference sums the actual FP32 inputs.
        // Future: compare FP32/FP64 accumulation and cancellation-heavy inputs.
        double reference = 0;
        for (int i = 0; i < N; ++i)
            reference += static_cast<double>(hMatrix[static_cast<size_t>(i) * N + i]);

        float *dMatrix = nullptr, *dA = nullptr, *dB = nullptr;
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dMatrix), bytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), firstBlocks * sizeof(float)));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), firstBlocks * sizeof(float)));
        CHECK(cudaMemcpy(dMatrix, hMatrix.data(), bytes, cudaMemcpyHostToDevice));

        // [H: WORK PER THREAD] Currently one input per thread at each stage.
        // Later: accumulate several inputs in a register before block reduction.
        // This can reduce block count, but too little parallel work can hurt latency hiding.
        // [I: CROSS-BLOCK REDUCTION] A block barrier cannot synchronize blocks.
        // Later kernels combine contiguous partial sums until only one remains.
        // Same-stream kernel ordering provides the stage dependencies;
        // no cudaDeviceSynchronize is required between these launches.
        auto launchTrace = [&]() -> float* {
            traceReduceSequential<<<static_cast<unsigned>(firstBlocks), threads, sharedBytes>>>(
                dMatrix, dA, N, static_cast<size_t>(N) + 1);
            CHECK(cudaGetLastError());

            size_t remaining = firstBlocks;
            float* current = dA;
            float* next = dB;
            while (remaining > 1) {
                size_t blocks = ceilDiv(remaining, threads);
                traceReduceSequential<<<static_cast<unsigned>(blocks), threads, sharedBytes>>>(
                    current, next, remaining, 1);
                CHECK(cudaGetLastError());
                remaining = blocks;
                std::swap(current, next);
            }
            return current;
        };

        int kernelsPerTrace = 1;
        for (size_t r = firstBlocks; r > 1; r = ceilDiv(r, threads))
            ++kernelsPerTrace;

        float* finalResult = nullptr;
        const int warmups = 10;
        const int repeats = 100;
        for (int i = 0; i < warmups; ++i)
            finalResult = launchTrace();
        CHECK(cudaDeviceSynchronize());

        // [J: TIMING] Time ALL reduction stages, not just the first kernel.
        // Includes GPU timeline gaps between launches; excludes allocation,
        // initialization, H2D/D2H copies and CPU validation.
        // A trace may need multiple kernels, so profiler launch-skip is NOT
        // automatically 10: warm-up launches = warmups * kernelsPerTrace.
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i)
            finalResult = launchTrace();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float totalMs = 0;
        CHECK(cudaEventElapsedTime(&totalMs, start, stop));
        float result = 0;
        CHECK(cudaMemcpy(&result, finalResult, sizeof(float), cudaMemcpyDeviceToHost));

        // Floating-point addition order changes: use tolerances, not exact equality.
        double absError = std::abs(static_cast<double>(result) - reference);
        double relError = reference != 0 ? absError / std::abs(reference) :
            (absError == 0 ? 0 : std::numeric_limits<double>::infinity());
        double tolerance = 1e-5 + 1e-5 * std::abs(reference);
        bool passed = std::isfinite(result) && absError <= tolerance;

        std::printf("GPU: %s\n", prop.name);
        std::printf("N=%d, threads/block=%d, first-stage blocks=%zu\n", N, threads, firstBlocks);
        std::printf("Kernels per trace: %d; warm-up kernel launches: %d\n",
                    kernelsPerTrace, warmups * kernelsPerTrace);
        std::printf("Shared memory/block: %zu bytes\n", sharedBytes);
        std::printf("GPU trace: %.9g; CPU FP64 reference: %.12g\n", result, reference);
        std::printf("Absolute error: %.9g; relative error: %.9g; tolerance: %.9g\n",
                    absError, relError, tolerance);
        std::printf("Average GPU trace time: %.6f ms (%d repetitions, all stages)\n",
                    totalMs / repeats, repeats);
        std::printf("Check: %s\n", passed ? "PASS" : "FAIL");

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        CHECK(cudaFree(dMatrix));
        CHECK(cudaFree(dA));
        CHECK(cudaFree(dB));
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return EXIT_FAILURE;
    }
}
