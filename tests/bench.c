/*
 * bench.c — MetaCuda micro-benchmarks (CUDA Driver API only).
 *   usage: bench <ptx-dir>
 * Reports launch overhead, synchronisation round trip, host<->device copy
 * bandwidth (unified memory), and kernel throughput for a few kernels.
 * Run tools/metal_baseline for the equivalent native-Metal numbers.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}
#define CK(x) do { CUresult _r = (x); if (_r) { fprintf(stderr, "%s failed: %d\n", #x, _r); exit(1); } } while (0)

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "build/ptx";
    char path[1024];
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    char name[256];
    CK(cuDeviceGetName(name, sizeof name, dev));
    CUmodule mb, mp;
    snprintf(path, sizeof path, "%s/basic.ptx", dir);
    double t0 = now_us();
    CK(cuModuleLoad(&mb, path));
    double t_load = now_us() - t0;
    snprintf(path, sizeof path, "%s/parallel.ptx", dir);
    CK(cuModuleLoad(&mp, path));
    CUfunction vecadd, saxpy, sgemm, reduce;
    CK(cuModuleGetFunction(&vecadd, mb, "vecadd"));
    CK(cuModuleGetFunction(&saxpy, mb, "saxpy"));
    CK(cuModuleGetFunction(&sgemm, mp, "sgemm_tiled"));
    CK(cuModuleGetFunction(&reduce, mp, "reduce_smem"));
    printf("MetaCuda benchmarks on %s\n", name);
    printf("  module load (basic.ptx, 8 kernels): %.2f ms  (run twice: 2nd run hits ~/.cache/metalcuda)\n", t_load / 1e3);

    /* launch overhead */
    int n = 4096;
    float one = 1.0f;
    CUdeviceptr x, y;
    CK(cuMemAlloc(&x, n * 4));
    CK(cuMemAlloc(&y, n * 4));
    void *a[] = {&n, &one, &x, &y};
    for (int i = 0; i < 100; i++) CK(cuLaunchKernel(saxpy, 16, 1, 1, 256, 1, 1, 0, 0, a, NULL));
    CK(cuCtxSynchronize());
    int iters = 20000;
    t0 = now_us();
    for (int i = 0; i < iters; i++) CK(cuLaunchKernel(saxpy, 16, 1, 1, 256, 1, 1, 0, 0, a, NULL));
    double t_enc = now_us() - t0;
    CK(cuCtxSynchronize());
    double t_all = now_us() - t0;
    printf("  cuLaunchKernel host cost        : %.2f us/launch\n", t_enc / iters);
    printf("  dependent launches, end to end  : %.2f us/launch (%d launches)\n", t_all / iters, iters);
    t0 = now_us();
    for (int i = 0; i < 200; i++) { CK(cuLaunchKernel(saxpy, 16, 1, 1, 256, 1, 1, 0, 0, a, NULL)); CK(cuCtxSynchronize()); }
    printf("  launch + cuCtxSynchronize       : %.1f us round trip\n", (now_us() - t0) / 200);

    /* copies */
    size_t bytes = 256u << 20;
    void *h = malloc(bytes);
    memset(h, 1, bytes);
    CUdeviceptr d;
    CK(cuMemAlloc(&d, bytes));
    CK(cuMemcpyHtoD(d, h, bytes));
    t0 = now_us();
    for (int i = 0; i < 5; i++) CK(cuMemcpyHtoD(d, h, bytes));
    double hd = 5.0 * bytes / ((now_us() - t0) * 1e-6) / 1e9;
    t0 = now_us();
    for (int i = 0; i < 5; i++) CK(cuMemcpyDtoH(h, d, bytes));
    double dh = 5.0 * bytes / ((now_us() - t0) * 1e-6) / 1e9;
    printf("  cuMemcpyHtoD / DtoH (256 MiB)   : %.1f / %.1f GB/s (CPU memcpy over unified memory)\n", hd, dh);

    /* kernels */
    int big = 1 << 24;
    CUdeviceptr A, B, C;
    CK(cuMemAlloc(&A, (size_t)big * 4));
    CK(cuMemAlloc(&B, (size_t)big * 4));
    CK(cuMemAlloc(&C, (size_t)big * 4));
    CK(cuMemsetD32(A, 0x3f800000u, big));
    CK(cuMemsetD32(B, 0x40000000u, big));
    void *va[] = {&A, &B, &C, &big};
    CUevent e0, e1;
    CK(cuEventCreate(&e0, 0));
    CK(cuEventCreate(&e1, 0));
    CK(cuLaunchKernel(vecadd, big / 256, 1, 1, 256, 1, 1, 0, 0, va, NULL));
    CK(cuEventRecord(e0, 0));
    for (int i = 0; i < 20; i++) CK(cuLaunchKernel(vecadd, big / 256, 1, 1, 256, 1, 1, 0, 0, va, NULL));
    CK(cuEventRecord(e1, 0));
    CK(cuEventSynchronize(e1));
    float ms;
    CK(cuEventElapsedTime(&ms, e0, e1));
    printf("  vecadd 16M floats               : %.3f ms/launch, %.0f GB/s effective\n", ms / 20, 3.0 * big * 4 * 20 / (ms * 1e-3) / 1e9);

    CUdeviceptr out;
    CK(cuMemAlloc(&out, 4));
    void *ra[] = {&A, &out, &big};
    CK(cuEventRecord(e0, 0));
    for (int i = 0; i < 20; i++) CK(cuLaunchKernel(reduce, big / 512, 1, 1, 256, 1, 1, 1024, 0, ra, NULL));
    CK(cuEventRecord(e1, 0));
    CK(cuEventSynchronize(e1));
    CK(cuEventElapsedTime(&ms, e0, e1));
    printf("  reduce_smem 16M floats          : %.3f ms/launch, %.0f GB/s\n", ms / 20, (double)big * 4 * 20 / (ms * 1e-3) / 1e9);

    int M = 1024, N = 1024, K = 1024;
    void *ga[] = {&A, &B, &C, &M, &N, &K};
    CK(cuLaunchKernel(sgemm, N / 16, M / 16, 1, 16, 16, 1, 0, 0, ga, NULL));
    CK(cuEventRecord(e0, 0));
    for (int i = 0; i < 10; i++) CK(cuLaunchKernel(sgemm, N / 16, M / 16, 1, 16, 16, 1, 0, 0, ga, NULL));
    CK(cuEventRecord(e1, 0));
    CK(cuEventSynchronize(e1));
    CK(cuEventElapsedTime(&ms, e0, e1));
    printf("  sgemm_tiled 1024^3 (naive 16x16): %.3f ms, %.0f GFLOP/s\n", ms / 10, 2.0 * M * N * K * 10 / (ms * 1e-3) / 1e9);
    return 0;
}
