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

## Files

- `matmul.cu`: all four implementations, timing, and correctness checks
