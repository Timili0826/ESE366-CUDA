// ESE366: Shared-memory tiled FP32 matrix multiplication, C = A * B.
// Build: nvcc -O3 -lineinfo multiplication_tiled.cu -o multiplication_tiled
// Run: ./multiplication_tiled [N=1000] [block_x=16] [block_y=16] [repeats=100]
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

template<int TILE>
__global__ void multiplicationTiled(const float* A, const float* B,
                                    float* C, int N)
{
    // [A: mapping] One block computes a TILE-by-TILE output region.
    // Each thread owns one output, as in the native implementation.
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + ty;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + tx;
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];
    float sum = 0.0f;
    const int phases = N / TILE + (N % TILE != 0);

    for (int phase = 0; phase < phases; ++phase) {
        const size_t aCol = static_cast<size_t>(phase) * TILE + tx;
        const size_t bRow = static_cast<size_t>(phase) * TILE + ty;
        // [B: boundary] Zero-fill partial tiles. Out-of-range output threads
        // must still participate in loading and both block-wide barriers.
        tileA[ty][tx] = (row < static_cast<size_t>(N) && aCol < static_cast<size_t>(N))
                     ? A[row * N + aCol] : 0.0f;
        tileB[ty][tx] = (bRow < static_cast<size_t>(N) && col < static_cast<size_t>(N))
                     ? B[bRow * N + col] : 0.0f;
        // [C: synchronization] Make both loaded tiles visible to all threads.
        __syncthreads();

        // [D: reuse] Each loaded A/B value serves TILE output dot products.
        // Source-level global loads fall via reuse; measured DRAM traffic
        // need not fall by TILE because the native kernel already uses caches.
        // No manual unroll directive: the compiler may optimize this fixed loop.
        for (int k = 0; k < TILE; ++k)
            sum += tileA[ty][k] * tileB[k][tx];

        // [E: synchronization] Complete all reads before reusing the buffers.
        // Keep both barriers for a simple, correct tile-loading loop.
        __syncthreads();
    }
    if (row < static_cast<size_t>(N) && col < static_cast<size_t>(N))
        C[row * N + col] = sum;

    // [F: tile-size tuning] TILE=16: 256 threads and 2048 shared bytes/block.
    // TILE=32: 1024 threads and 8192 shared bytes/block.
    // Larger tiles offer more reuse and fewer phases but different residency.
    // [G: precision/unrolling] Keep FP32 and one output/thread for comparison.
    // An inner-loop unroll directive alone does not remove phase barriers.
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
            throw std::invalid_argument("Usage: ./multiplication_tiled [N] [block_x] [block_y] [repeats]");
        int N = argc > 1 ? positiveInt(argv[1]) : 1000;
        int bx = argc > 2 ? positiveInt(argv[2]) : 16;
        int by = argc > 3 ? positiveInt(argv[3]) : 16;
        int repeats = argc > 4 ? positiveInt(argv[4]) : 100;
        if (bx != by || (bx != 16 && bx != 32))
            throw std::invalid_argument("Tiled version requires block=(16,16) or (32,32).");
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
        size_t sharedBytes = 2 * static_cast<size_t>(bx) * by * sizeof(float);
        if (sharedBytes > prop.sharedMemPerBlock)
            throw std::invalid_argument("Shared-memory allocation exceeds device limit.");
        // [H: tile-size experiment] Square tile and block sizes match.
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

        // Same input, timing and validation as native; select a compiled tile.
        auto launch = [&]() {
            if (bx == 16)
                multiplicationTiled<16><<<grid, block>>>(dA, dB, dC, N);
            else
                multiplicationTiled<32><<<grid, block>>>(dA, dB, dC, N);
        };

        const int warmups = 10;
        for (int i = 0; i < warmups; ++i)
            launch();
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // [H: timing] Match earlier operations: event-timed batch / repetitions.
        // Exclude allocation, copies, initialization, and CPU validation.
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i)
            launch();
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
        std::printf("Warps/block=%zu, launched warps=%zu; static shared memory/block=%zu bytes\n",
                    warpsPerBlock, blocks * warpsPerBlock, sharedBytes);
        std::printf("Tile=%d; phases=%d; source barriers/block=%d per kernel\n",
                    bx, N / bx + (N % bx != 0), 2 * (N / bx + (N % bx != 0)));
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
