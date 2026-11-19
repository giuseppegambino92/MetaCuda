// Cooperative kernels: shared memory, barriers, warp intrinsics, atomics.
#include <cuda_runtime.h>

// Tree reduction in dynamic shared memory + one float atomicAdd per block.
extern "C" __global__ void reduce_smem(const float *in, float *out, int n) {
    extern __shared__ float s[];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + t;
    float v = 0.0f;
    if (i < n) v = in[i];
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    s[t] = v;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (t < k) s[t] += s[t + k];
        __syncthreads();
    }
    if (t == 0) atomicAdd(out, s[0]);
}

// Warp-shuffle reduction: per-warp partial sums via __shfl_down_sync.
extern "C" __global__ void reduce_warp(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = i < n ? in[i] : 0;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    __shared__ int warp_sums[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        int w = lane < nw ? warp_sums[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) w += __shfl_xor_sync(0xffffffffu, w, off);
        if (lane == 0) atomicAdd(out, w);
    }
}

// Votes and shuffles with divergent producers.
extern "C" __global__ void warp_vote(const int *in, unsigned *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int v = in[i];
    unsigned ballot = __ballot_sync(0xffffffffu, v > 0);
    int any = __any_sync(0xffffffffu, v > 100);
    int all = __all_sync(0xffffffffu, v > -1000);
    int bcast = __shfl_sync(0xffffffffu, v, 7);
    int up = __shfl_up_sync(0xffffffffu, v, 3);
    float fx = __shfl_xor_sync(0xffffffffu, (float)v * 0.5f, 5);
    out[i * 6 + 0] = ballot;
    out[i * 6 + 1] = (unsigned)any | ((unsigned)all << 1);
    out[i * 6 + 2] = (unsigned)bcast;
    out[i * 6 + 3] = (unsigned)up;
    out[i * 6 + 4] = __float_as_uint(fx);
    out[i * 6 + 5] = (unsigned)lane;
}

// Shared-memory histogram with integer atomics on shared and global memory.
extern "C" __global__ void histogram(const unsigned char *data, unsigned *bins, int n) {
    __shared__ unsigned local[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) atomicAdd(&local[data[i]], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x)
        if (local[b]) atomicAdd(&bins[b], local[b]);
}

// Block-wide inclusive scan (Hillis–Steele), barriers inside a loop.
extern "C" __global__ void scan_block(const int *in, int *out, int n) {
    __shared__ int buf[2][1024];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    int src = 0;
    buf[0][t] = i < n ? in[i] : 0;
    __syncthreads();
    for (int off = 1; off < blockDim.x; off <<= 1) {
        int v = buf[src][t];
        if (t >= off) v += buf[src][t - off];
        buf[src ^ 1][t] = v;
        src ^= 1;
        __syncthreads();
    }
    if (i < n) out[i] = buf[src][t];
}

// Tiled SGEMM with static shared memory: C = A(MxK) * B(KxN).
#define TILE 16
extern "C" __global__ void sgemm_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[threadIdx.y][threadIdx.x] = (row < M && k0 + threadIdx.x < K) ? A[row * K + k0 + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && k0 + threadIdx.y < K) ? B[(k0 + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// Assorted atomics: CAS loop (float max), exch, min/max, inc wrap, 64-bit add.
extern "C" __global__ void atomics(const float *in, int n, float *fmax_out, int *imax, int *imin, unsigned *inc, unsigned long long *sum64, int *exch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    // float max via CAS
    int *addr = (int *)fmax_out;
    int old = *addr, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= v) break;
        old = atomicCAS(addr, assumed, __float_as_int(v));
    } while (old != assumed);
    atomicMax(imax, (int)(v * 100.0f));
    atomicMin(imin, (int)(v * 100.0f));
    atomicInc(inc, 99u);
    atomicAdd(sum64, (unsigned long long)i * 1000003ull);
    atomicExch(exch, i);
}

// Per-thread array with dynamic indexing → PTX local memory (.local depot).
extern "C" __global__ void local_array(const int *idx, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    volatile int arr[24];
    for (int k = 0; k < 24; k++) arr[k] = k * k + i;
    int acc = 0;
    for (int k = 0; k < 8; k++) acc += arr[(idx[i] + k * 5) % 24];
    out[i] = acc;
}

// __syncthreads_count / block-level predicates (bar.red).
extern "C" __global__ void sync_count(const int *in, int *out) {
    int v = in[blockIdx.x * blockDim.x + threadIdx.x];
    int c = __syncthreads_count(v & 1);
    int a = __syncthreads_and(v >= 0);
    int o = __syncthreads_or(v == 42);
    if (threadIdx.x == 0) { out[blockIdx.x * 3] = c; out[blockIdx.x * 3 + 1] = a; out[blockIdx.x * 3 + 2] = o; }
}

// Loop with a per-thread trip count followed by a barrier and a warp shuffle.
// The loop is entered unconditionally, so its exit block has a single
// predecessor: a naive structurizer would inline the barrier *inside* the loop.
extern "C" __global__ void loop_then_barrier(int *out) {
    __shared__ int s[256];
    int t = threadIdx.x;
    int i = 0, acc = 0;
    do {
        acc = acc * 1664525 + 1013904223 + i; // LCG step: no closed form, the loop survives -O3
        i++;
    } while (i < (t % 37) + 1);
    s[t] = acc;
    __syncthreads();
    int nb = s[(t + 1) % blockDim.x];
    int w = __shfl_xor_sync(0xffffffffu, acc, 1);
    out[blockIdx.x * blockDim.x + t] = nb * 1000 + w;
}
