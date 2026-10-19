// Baseline: out[col][row] = in[row][col], separate input/output arrays.
// Build: nvcc -O3 -lineinfo transpose.cu -o transpose
// Run: ./transpose [N=1024] [block_x=16] [block_y=16]
#include <cuda_runtime.h>
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

__global__ void transposeNaive(const float* in, float* out, int N)
{
    // [A] One thread reads one element at (row, col).
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // [B] Boundary guard: keep it for non-divisible dimensions.
    if (row < N && col < N) {
        // [C] Adjacent x threads read adjacent input values,
        // but their output addresses are N elements apart.
        // Later: use a shared-memory tile to reorganize the writes.
        out[static_cast<size_t>(col) * N + row] =
            in[static_cast<size_t>(row) * N + col];
    }
    // [D] No cross-thread data dependency here: no __syncthreads needed.
    // [E] No loop here to unroll. Test unrolling in a later tiled version.
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
        if (argc > 4) throw std::invalid_argument("Usage: ./transpose [N] [block_x] [block_y]");
        int N  = argc > 1 ? positiveInt(argv[1]) : 1024;
        int bx = argc > 2 ? positiveInt(argv[2]) : 16;
        int by = argc > 3 ? positiveInt(argv[3]) : 16;
        if (static_cast<size_t>(N) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(N) / sizeof(float))
            throw std::invalid_argument("Matrix byte size overflows size_t.");
        size_t count = static_cast<size_t>(N) * N;
        size_t bytes = count * sizeof(float);

        cudaDeviceProp prop{};
        CHECK(cudaGetDeviceProperties(&prop, 0));
        if (bx > prop.maxThreadsDim[0] || by > prop.maxThreadsDim[1] ||
            static_cast<long long>(bx) * by > prop.maxThreadsPerBlock)
            throw std::invalid_argument("Block dimensions exceed device limits.");

        // [F] Vary block shape; recompute grid to cover the entire matrix.
        dim3 block(bx, by);
        dim3 grid(N / bx + (N % bx != 0), N / by + (N % by != 0));
        if (grid.x > static_cast<unsigned>(prop.maxGridSize[0]) ||
            grid.y > static_cast<unsigned>(prop.maxGridSize[1]))
            throw std::invalid_argument("Grid dimensions exceed device limits.");

        std::vector<float> hIn(count), hOut(count);
        for (size_t i = 0; i < count; ++i)
            hIn[i] = static_cast<float>(i % 1009);

        float *dIn = nullptr, *dOut = nullptr;
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dIn), bytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&dOut), bytes));
        CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dOut, 0xff, bytes));

        for (int i = 0; i < 10; ++i)
            transposeNaive<<<grid, block>>>(dIn, dOut, N);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // [G] GPU kernel timing only: excludes allocation, copies and checking.
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int repeats = 100;
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i)
            transposeNaive<<<grid, block>>>(dIn, dOut, N);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float totalMs = 0;
        CHECK(cudaEventElapsedTime(&totalMs, start, stop));
        CHECK(cudaMemcpy(hOut.data(), dOut, bytes, cudaMemcpyDeviceToHost));

        // Transpose only copies values, so exact equality is appropriate here.
        size_t mismatches = 0;
        for (int row = 0; row < N; ++row)
            for (int col = 0; col < N; ++col)
                if (hOut[static_cast<size_t>(col) * N + row] !=
                    hIn[static_cast<size_t>(row) * N + col])
                    ++mismatches;

        std::printf("GPU: %s\n", prop.name);
        std::printf("N=%d, block=(%u,%u), grid=(%u,%u)\n",
                    N, block.x, block.y, grid.x, grid.y);
        std::printf("Average kernel time: %.6f ms (%d repetitions)\n",
                    totalMs / repeats, repeats);
        std::printf("Check: %s, mismatches=%zu\n",
                    mismatches == 0 ? "PASS" : "FAIL", mismatches);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        CHECK(cudaFree(dIn));
        CHECK(cudaFree(dOut));
        return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return EXIT_FAILURE;
    }
}
