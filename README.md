# Matrix Multiplication: CPU vs CUDA

Square matrix multiplication (`C = A * B`, float32) implemented four ways in CUDA C++ and benchmarked on a Colab T4:

1. Serial CPU
2. Multithreaded CPU (`std::thread`, 4 threads, rows split between them)
3. CUDA naive (one thread per output element)
4. CUDA tiled (16x16 shared-memory tiles)

Every version is checked against the serial result.

## Build and run

```bash
nvcc -O2 -arch=sm_75 matmul.cu -o matmul
./matmul 1024
```

`sm_75` targets the T4. Change it for a different GPU. The matrix size is the only argument (default 1024).

## Results

Colab T4. GPU times are kernel-only, so host/device copies are not included. The serial baseline uses the cache-friendly `i-k-j` loop order (see below).

| N | Serial CPU (ms) | Threaded CPU (ms) | CUDA naive (ms) | CUDA tiled (ms) |
|------|---------|---------|-------|-------|
| 128  | 2.23    | 1.94    | 0.03  | 0.03  |
| 512  | 100.83  | 97.08   | 0.59  | 0.40  |
| 1024 | 750.49  | 765.99  | 4.28  | 2.69  |
| 2048 | 6021.90 | 5969.47 | 45.16 | 27.19 |

Max difference against the serial result was at most 1.8e-4 for every size, so all versions agree.

Throughput, counting 2N³ floating-point operations:

| N | Serial CPU | CUDA naive | CUDA tiled |
|------|-----------|-----------|-----------|
| 1024 | ~2.9 GFLOP/s | ~500 GFLOP/s | ~800 GFLOP/s |
| 2048 | ~2.9 GFLOP/s | ~380 GFLOP/s | ~630 GFLOP/s |

A T4's FP32 peak is roughly 8 TFLOP/s, so the tiled kernel reaches around 10% of it.

## What I found

- **CPU loop order mattered a lot.** My first version used the textbook `i-j-k` order, which reads `B` down a column and misses cache constantly. Switching to `i-k-j` made the serial version about 11x faster at N=2048 (66 s down to 6 s) and about 4x faster at N=1024, with identical results. That also shrank the GPU speedup from ~2500x to ~220x, which is a fairer comparison.
- **Tiling helps, but modestly.** The tiled kernel is about 1.5 to 1.7x faster than naive at N=512 and above. Each block loads a tile into shared memory once and reuses it 16 times, cutting global memory traffic. A simple 16x16 tile still leaves most of the hardware idle.
- **Timings vary between runs.** The naive kernel at N=2048 took 42 to 50 ms across my runs, so treat the ratios as approximate.
- **Small sizes aren't meaningful.** At N=128 both kernels finish in a few hundredths of a millisecond, so launch overhead and timer noise dominate.
- **Threads gave no speedup at larger sizes.** The threaded version (4 threads) was about 1.0x at N=1024 and 2048. Colab's CPU has very few cores, which probably explains it, and I did not confirm the exact core count. (Edit this line after running `nproc`.)

## Limitations

- GPU timings leave out the host/device copies.
- Each number is a single run, not an average.
- Only 16x16 tiles were tested.
- The thread count is hardcoded to 4 (`NUM_THREADS`), not matched to the machine.

## Possible next steps

- Try 32x32 tiles, or have each thread compute several outputs (register blocking).
- Compare against cuBLAS.
- Include copy time and average several runs.
- Test threading on a machine with more cores.

## Files

- `matmul.cu`: all four implementations, timing, and correctness checks
