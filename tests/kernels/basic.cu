// Element-wise kernels: arithmetic, conversions, bit manipulation, math, structs, f64.
#include <cuda_runtime.h>

extern "C" __global__ void vecadd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void saxpy(int n, float alpha, const float *__restrict__ x, float *__restrict__ y) {
    // grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) y[i] = alpha * x[i] + y[i];
}

extern "C" __global__ void int_ops(const unsigned *in, unsigned *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = in[i];
    int s = (int)x;
    unsigned *o = out + 16 * i;
    o[0] = __popc(x);
    o[1] = __clz((int)x);
    o[2] = __brev(x);
    o[3] = __byte_perm(x, 0x12345678u, 0x3210 + (i & 3) * 0x1111);
    o[4] = __funnelshift_l(x, x ^ 0xdeadbeefu, i);
    o[5] = __umulhi(x, 0x9e3779b9u);
    o[6] = (unsigned)(s / (i - 7 == 0 ? 3 : i - 7));
    o[7] = (unsigned)(s % (i + 3));
    o[8] = (unsigned)(s >> (i & 31));
    o[9] = x >> (i & 31);
    o[10] = x << (i & 31);
    o[11] = (unsigned)min(s, -s);
    o[12] = max(x, 1000000u * (unsigned)i);
    o[13] = (x & 0xffu) * (x >> 24) + (unsigned)__mul24(s, 3);
    o[14] = (unsigned)(((unsigned long long)x * 0x100000001ull) >> 17);
    o[15] = (unsigned)__ffs((int)x);
}

extern "C" __global__ void float_ops(const float *in, float *out, int *iout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    float *o = out + 8 * i;
    int *io = iout + 4 * i;
    o[0] = sqrtf(fabsf(x));
    o[1] = fmaf(x, 1.5f, -2.25f);
    o[2] = fminf(x, 0.5f) + fmaxf(x, -0.5f);
    o[3] = x / 3.0f;
    o[4] = floorf(x) + ceilf(x * 0.5f);
    o[5] = (float)(int)x;           // truncation
    o[6] = __expf(x * 0.1f);         // approximate
    o[7] = rsqrtf(fabsf(x) + 1.0f);  // approximate
    io[0] = __float2int_rn(x * 1000.0f);
    io[1] = (int)(x * 1e9f);         // saturating conversion
    io[2] = x > 0.0f ? 1 : (x < 0.0f ? -1 : 0);
    io[3] = (int)(unsigned char)(unsigned)(x * 10.0f);
}

struct Params {
    float scale;
    int offset;
    float bias[2];
    unsigned long long mask;
};

extern "C" __global__ void struct_param(Params p, const int *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)((in[i] + p.offset) & (int)p.mask) * p.scale + p.bias[i & 1];
}

extern "C" __global__ void f64_ops(const double *a, const double *b, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i] + 0.5 * a[i] - sqrt(fabs(b[i]));
}

extern "C" __global__ void grid2d(float *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    if (x < w && y < h) out[(z * h + y) * w + x] = (float)(x + 1000 * y + 1000000 * z);
}

extern "C" __global__ void small_types(const unsigned char *u8, const short *s16, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (int)u8[i] * 3 + (int)s16[i] - (signed char)u8[i];
}
