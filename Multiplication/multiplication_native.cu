// ESE366: FP32 matrix multiplication baseline, C = A * B.
// Build: nvcc -O3 -lineinfo multiplication_native.cu -o multiplication_native
// Run: ./multiplication_native [N=1000] [block_x=16] [block_y=16] [repeats=100]
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <set>
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

__global__ void multiplicationNative(const float* A, const float* B,
                                     float* C, int N)
{
    // [A: mapping] One thread computes one output element (row, col).
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < static_cast<size_t>(N) && col < static_cast<size_t>(N)) {
        // [B: registers] Accumulate the dot product in a thread-local scalar.
        // The compiler determines actual register usage. Default nvcc may use FMA.
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            // [C: global memory] Nearby x threads share an A element and
            // access adjacent B columns at a given k. One thread walks B
            // with stride N, but across x threads the B accesses are adjacent.
            // Many outputs reuse A/B values; caches may capture some reuse.
            // Later: shared-memory tiles can explicitly reuse both inputs.
            sum += A[row * N + k] * B[static_cast<size_t>(k) * N + col];
        }
        C[row * N + col] = sum;
    }
    // [D: synchronization] This baseline has no cross-thread dependencies
    // and needs no __syncthreads(). A tiled version needs barriers before
    // reading a loaded tile and before overwriting it for the next phase.
    // All threads must reach those barriers; zero-fill invalid tile entries.
    // [E: unrolling] No manual loop expansion is requested here.
    // Later: expand a fixed-size tile's inner k loop. This alone does not
    // reduce the number of tile-loading phases or their required barriers.
    // [F: precision] Keep FP32 accumulation for controlled kernel comparisons.
    // A separate FP64 accumulation experiment can measure time/accuracy costs.
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
        if (argc > 5)
            throw std::invalid_argument("Usage: ./multiplication_native [N] [block_x] [block_y] [repeats]");
        int N = argc > 1 ? positiveInt(argv[1]) : 1000;
        int bx = argc > 2 ? positiveInt(argv[2]) : 16;
        int by = argc > 3 ? positiveInt(argv[3]) : 16;
        int repeats = argc > 4 ? positiveInt(argv[4]) : 100;
        size_t n = static_cast<size_t>(N);
        if (n > std::numeric_limits<size_t>::max() / n / sizeof(float))
            throw std::invalid_argument("Matrix byte size overflows size_t.");
        size_t count = n * n;
        size_t bytes = count * sizeof(float);

        cudaDeviceProp prop{};
        CHECK(cudaGetDeviceProperties(&prop, 0));
        if (bx > prop.maxThreadsDim[0] || by > prop.maxThreadsDim[1] ||
            static_cast<long long>(bx) * by > prop.maxThreadsPerBlock)
            throw std::invalid_argument("Block dimensions exceed device limits.");
        // [G: launch tuning] Compare block shapes with fixed matrix/input data.
        dim3 block(bx, by);
        dim3 grid(N / bx + (N % bx != 0), N / by + (N % by != 0));
        if (grid.x > static_cast<unsigned>(prop.maxGridSize[0]) ||
            grid.y > static_cast<unsigned>(prop.maxGridSize[1]))
            throw std::invalid_argument("Grid dimensions exceed device limits.");

        std::vector<float> hA(count), hB(count), hC(count);
        for (size_t i = 0; i < count; ++i) {
            // Deterministic signed fractional FP32 inputs, identical across versions.
            hA[i] = (static_cast<int>(i % 2001) - 1000) / 1000.0f;
            hB[i] = (static_cast<int>(((i % 1999) * 17) % 1999) - 999) / 1000.0f;
        }
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), bytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), bytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dC), bytes));
        CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
        // NaN sentinel helps detect output elements that were never written.
        CHECK(cudaMemset(dC, 0xff, bytes));

        const int warmups = 10;
        for (int i = 0; i < warmups; ++i)
            multiplicationNative<<<grid, block>>>(dA, dB, dC, N);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // [H: timing] Match earlier operations: event-timed batch / repetitions.
        // Exclude allocation, copies, initialization, and CPU validation.
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i)
            multiplicationNative<<<grid, block>>>(dA, dB, dC, N);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        double averageMs = static_cast<double>(elapsed) / repeats;
        CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

        // [I: validation] All outputs must be finite. Full FP64 reference for
        // N<=128; for larger N, check 1024 distinct deterministic output positions.
        // Sampling includes matrix corners and the middle flattened position.
        // A sampled PASS is not a full numerical verification of a large matrix.
        size_t nonfinite = 0;
        for (float value : hC) if (!std::isfinite(value)) ++nonfinite;
        bool fullReference = N <= 128;
        std::vector<size_t> positions;
        if (fullReference) {
            positions.reserve(count);
            for (size_t i = 0; i < count; ++i) positions.push_back(i);
        } else {
            std::set<size_t> chosen{0, n - 1, (n - 1) * n, count - 1, count / 2};
            std::mt19937_64 rng(366);
            while (chosen.size() < 1024) chosen.insert(rng() % count);
            positions.assign(chosen.begin(), chosen.end());
        }
        const double atol = 1e-4, rtol = 1e-4;
        double maxAbs = 0.0, maxRef = 0.0;
        size_t mismatches = 0;
        for (size_t index : positions) {
            size_t row = index / n, col = index % n;
            double ref = 0.0;
            for (int k = 0; k < N; ++k)
                ref += static_cast<double>(hA[row * n + k])
                     * static_cast<double>(hB[static_cast<size_t>(k) * n + col]);
            double error = std::abs(static_cast<double>(hC[index]) - ref);
            maxRef = std::max(maxRef, std::abs(ref));
            if (!std::isfinite(hC[index])) {
                maxAbs = std::numeric_limits<double>::infinity();
                ++mismatches;
            } else {
                maxAbs = std::max(maxAbs, error);
                if (error > atol + rtol * std::abs(ref)) ++mismatches;
            }
        }
        double normError = maxRef > 0.0 ? maxAbs / maxRef : maxAbs;
        // [J: throughput] Conventional 2*N^3 useful FLOPs / measured time.
        // No claim is made about achieved DRAM bandwidth or hardware peak FLOPs.
        double gflops = 2.0 * n * n * n / (averageMs * 1e6);
        size_t threadsPerBlock = static_cast<size_t>(bx) * by;
        size_t blocks = static_cast<size_t>(grid.x) * grid.y;
        size_t warpsPerBlock = (threadsPerBlock + 31) / 32;
        bool passed = mismatches == 0 && nonfinite == 0;
        std::printf("GPU: %s\n", prop.name);
        std::printf("N=%d, block=(%d,%d), grid=(%u,%u)\n", N, bx, by, grid.x, grid.y);
        std::printf("Threads/block=%zu, grid blocks=%zu, launched threads=%zu\n",
                    threadsPerBlock, blocks, blocks * threadsPerBlock);
        std::printf("Warps/block=%zu, launched warps=%zu; explicit shared memory/block=0 bytes\n",
                    warpsPerBlock, blocks * warpsPerBlock);
        std::printf("Warm-up kernel launches: %d\n", warmups);
        std::printf("Average kernel time: %.6f ms (%d repetitions)\n", averageMs, repeats);
        std::printf("Effective compute throughput: %.3f GFLOP/s (2*N^3/time)\n", gflops);
        std::printf("Validation: all %zu outputs checked for finiteness; %s CPU FP64 reference, %zu/%zu outputs\n",
                    count, fullReference ? "FULL" : "SAMPLED", positions.size(), count);
        std::printf("%s max absolute error: %.9g; normalized infinity error: %.9g\n",
                    fullReference ? "Full-output" : "Sampled", maxAbs, normError);
        std::printf("Tolerance per checked element: %.1e + %.1e * abs(reference)\n", atol, rtol);
        std::printf("Check: %s, reference mismatches=%zu, nonfinite outputs=%zu\n",
                    passed ? "PASS" : "FAIL", mismatches, nonfinite);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        CHECK(cudaFree(dA));
        CHECK(cudaFree(dB));
        CHECK(cudaFree(dC));
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return EXIT_FAILURE;
    }
}
