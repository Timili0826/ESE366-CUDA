# CUDA Matrix Operations on NVIDIA Jetson Orin

An ESE 366 course project implementing and optimizing four matrix operations in CUDA: **transpose, trace, addition, and multiplication**.

The project compares baseline kernels with changes to memory access, block configuration, shared-memory layout, and work per thread. CUDA-event timing measures performance, while NVIDIA Nsight Compute helps explain execution behavior.

## Test Environment

- **GPU:** NVIDIA Jetson Orin, compute capability 8.7
- **Language:** CUDA C++
- **Compiler flags:** `nvcc -O3 -lineinfo`
- **GPU arithmetic/data:** FP32
- **Main matrix sizes:** 1000 × 1000 and 4096 × 4096

## Implementations

| Operation | Implementations | Retained configuration |
| --- | --- | --- |
| Transpose | [Native](Transpose/transpose_native.cu), [tiled](Transpose/transpose_tiled.cu), [padded](Transpose/transpose_padded.cu) | Padded tile, 16 × 16 threads |
| Trace | [Native](Trace/trace_native.cu), [sequential](Trace/trace_sequential.cu), [pair-load](Trace/trace_pairload.cu) | Sequential reduction, 128 threads/block |
| Addition | [Native](Addition/addition_native.cu), [unrolled](Addition/addition_unrolled.cu) | Up to four elements/thread, 128 threads/block |
| Multiplication | [Native](Multiplication/multiplication_native.cu), [tiled](Multiplication/multiplication_tiled.cu) | Shared-memory tile, 16 × 16 threads |

These configurations were selected for the tested workloads; they are not claimed to be optimal for every GPU or matrix size.

## Results

Representative controlled comparisons at **N = 4096**:

| Operation | Comparison | Baseline (ms) | Optimized (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| Transpose | Native → padded, both 16 × 16 | 4.216940 | 2.277422 | 1.85× |
| Trace | Native → sequential, both 256 threads/block | 0.014928 | 0.012377 | 1.21× |
| Addition | Native → four-element unrolled, both 128 threads/block | 2.217070 | 2.108076 | 1.05× |
| Multiplication | Native → tiled, both 16 × 16 | 908.577813 | 620.536289 | 1.46× |

- Each value is the **median of three runs**, with each run averaging **100 operations** after 10 warm-up operations.
- Trace timing includes both reduction stages; each trace uses two kernel launches at these sizes.
- Allocation, initialization, host–device transfers, and CPU validation are excluded.
- The trace row fixes 256 threads/block to isolate the reduction change. A separate block-size experiment selected 128 threads/block for subsequent tests.
- Speedups compare versions of the same operation. Raw times do not rank efficiency across operations with different amounts of work.
- Timings printed while profiling are excluded from this table.

### Main Findings

- **Transpose:** Shared-memory tiling improved write access organization. Padding added a smaller benefit for only 64 extra shared-memory bytes per block.
- **Trace:** Sequential reduction reduced execution work. Pair-load reduced the number of blocks but showed no clear additional timing benefit.
- **Addition:** Processing four elements per thread with loop expansion improved time modestly while retaining contiguous accesses. This changes work distribution as well as loop organization.
- **Multiplication:** Tiled 16 outperformed the native kernel. Tiled 32 reduced load work and synchronization phases further, but lower occupancy accompanied an approximately 28% slowdown relative to tiled 16.

## Build and Run

Run the following from the repository root on a CUDA-capable Jetson with `nvcc` installed. Each source file is a standalone program. These commands build the retained implementations:

```bash
mkdir -p build

nvcc -O3 -lineinfo Transpose/transpose_padded.cu -o build/transpose_padded
nvcc -O3 -lineinfo Trace/trace_sequential.cu -o build/trace_sequential
nvcc -O3 -lineinfo Addition/addition_unrolled.cu -o build/addition_unrolled
nvcc -O3 -lineinfo Multiplication/multiplication_tiled.cu -o build/multiplication_tiled
```

Example runs:

```bash
./build/transpose_padded 4096 16 16
./build/trace_sequential 4096 128
./build/addition_unrolled 4096 128
./build/multiplication_tiled 4096 16 16
```

Repeat each command three times to follow the reported timing procedure. Large multiplication tests can take minutes because each run includes 10 warm-ups and 100 timed launches.

Multiplication accepts an optional final repetition count:

```bash
./build/multiplication_tiled 4096 16 16 5
```

This shortens a trial run; the published ordinary timing results used 100 repetitions.

## Correctness and Profiling

- **Transpose:** Every output is checked against the expected rearrangement.
- **Trace and addition:** Numerical results are checked against CPU FP64 references formed from the stored FP32 inputs.
- **Multiplication:** Every output is checked for finiteness. For N ≤ 128, every output receives a CPU FP64 reference check; larger matrices use 1024 fixed sample positions.

For example, the following exercises full numerical validation and partial-tile handling:

```bash
./build/multiplication_tiled 127 16 16
```

All reported checks passed. A sampled multiplication check does not establish numerical correctness for every entry of a large matrix.

Nsight Compute profiles examine launch configuration, occupancy, memory requests, instruction work, and sampled stalls. The report distinguishes L1/TEX and L2 counters from DRAM traffic and separates profiling overhead from ordinary timing.

The experiments did not establish the largest supported matrix size, GPU FP64 performance, or energy consumption.

## Full Report

[Read the full experimental report](reports/ESE366_CUDA_Report.pdf) for mathematical models, kernel design, results at both matrix sizes, profiling analysis, and references.

## Technical References

- NVIDIA, [CUDA C++ Programming Guide, CUDA 12.6](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)
- NVIDIA, [CUDA C++ Best Practices Guide, CUDA 12.6](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-best-practices-guide/index.html)
- Mark Harris, [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- Mark Harris, [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- NVIDIA, [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- NVIDIA, [Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://docs.nvidia.com/cuda/archive/12.6.0/floating-point/index.html)
