// 64-bit atomics and IEEE binary64 arithmetic.
#include <cuda_runtime.h>

// Every thread must receive a distinct old value: the returned values are a
// permutation of start, start+inc, ... (the counter crosses a 32-bit boundary).
extern "C" __global__ void atom64_add_unique(unsigned long long *ctr, unsigned long long *olds, unsigned long long inc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) olds[i] = atomicAdd(ctr, inc);
}

// 64-bit CAS loop increment + exchange conservation + min/max with results.
extern "C" __global__ void atom64_misc(unsigned long long *cas_ctr, unsigned long long *xchg, unsigned long long *xchg_out,
                                       unsigned long long *mx, unsigned long long *mn, unsigned long long *mx_out, long long *smx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long old = *cas_ctr, assumed;
    do {
        assumed = old;
        old = atomicCAS(cas_ctr, assumed, assumed + 0x100000001ull);
    } while (old != assumed);
    unsigned long long v = 0x123456789ull * (unsigned long long)(i + 1);
    xchg_out[i] = atomicExch(xchg, v);
    mx_out[i] = atomicMax(mx, v ^ 0xF0F0F0F0F0F0ull);
    atomicMin(mn, v ^ 0xF0F0F0F0F0F0ull);
    atomicMax(smx, (long long)(v ^ 0x8000000000000000ull));
}

// Same in shared memory: per-block 64-bit counter, distinct old values.
extern "C" __global__ void atom64_shared(unsigned long long *olds, unsigned long long *totals) {
    __shared__ unsigned long long c;
    if (threadIdx.x == 0) c = 0xFFFFFFF0ull;
    __syncthreads();
    olds[blockIdx.x * blockDim.x + threadIdx.x] = atomicAdd(&c, 1ull);
    __syncthreads();
    if (threadIdx.x == 0) totals[blockIdx.x] = c;
}

// double atomicAdd: sums of exactly representable values must be exact.
extern "C" __global__ void atom_f64_add(const double *v, double *sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(sum, v[i]);
}

// One IEEE operation per output so the CPU can reproduce every result bit for bit.
extern "C" __global__ void f64_exact(const double *a, const double *b, const double *c, unsigned long long *o, int *io, float *fo, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = a[i], y = b[i], z = c[i];
    unsigned long long *r = o + 12 * i;
    r[0] = __double_as_longlong(x + y);
    r[1] = __double_as_longlong(x - y);
    r[2] = __double_as_longlong(x * y);
    r[3] = __double_as_longlong(x / y);
    r[4] = __double_as_longlong(sqrt(x));
    r[5] = __double_as_longlong(fma(x, y, z));
    r[6] = __double_as_longlong(fmin(x, y));
    r[7] = __double_as_longlong(fmax(x, y));
    r[8] = __double_as_longlong((double)(long long)i * 0.1);
    r[9] = __double_as_longlong(floor(x) + ceil(y));
    r[10] = __double_as_longlong((double)(float)z);
    r[11] = __double_as_longlong(rint(x));
    io[4 * i + 0] = (x < y) | ((x == y) << 1) | ((x > y) << 2) | ((x != x) << 3);
    io[4 * i + 1] = (int)fmin(fmax(x, -2e9), 2e9);
    io[4 * i + 2] = __double2int_rn(y);
    io[4 * i + 3] = (int)(long long)fmin(fmax(z, -9e18), 9e18);
    fo[i] = (float)x;
}
