/*
 * test_driver_suite.c — CUDA Driver API conformance tests for MetaCuda.
 *
 * Every kernel comes from the tests/kernels .cu files, compiled to PTX by clang's NVPTX
 * backend (real compiler output). Each result is checked against a CPU
 * reference implementation.
 *
 *   usage: test_driver_suite <ptx-dir>
 */
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_fail = 0, g_pass = 0;
static const char *g_dir = "build/ptx";

#define CK(call)                                                                                      \
    do {                                                                                              \
        CUresult _r = (call);                                                                         \
        if (_r != CUDA_SUCCESS) {                                                                     \
            const char *_s = NULL;                                                                    \
            cuGetErrorName(_r, &_s);                                                                  \
            fprintf(stderr, "  CUDA error %s (%d) at %s:%d: %s\n", _s ? _s : "?", _r, __FILE__, __LINE__, #call); \
            return 0;                                                                                 \
        }                                                                                             \
    } while (0)

#define EXPECT(cond, ...)                                       \
    do {                                                        \
        if (!(cond)) {                                          \
            fprintf(stderr, "  check failed: " __VA_ARGS__);    \
            fprintf(stderr, "\n");                              \
            return 0;                                           \
        }                                                       \
    } while (0)

static uint32_t rng_state = 12345;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static float rndf(float lo, float hi) { return lo + (hi - lo) * (float)(rnd() & 0xFFFFFF) / 16777216.0f; }

static CUmodule load(const char *name) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.ptx", g_dir, name);
    CUmodule m = NULL;
    CUresult r = cuModuleLoad(&m, path);
    if (r) { fprintf(stderr, "  cuModuleLoad(%s) failed: %d\n", path, r); return NULL; }
    return m;
}
static CUfunction fn(CUmodule m, const char *name) {
    CUfunction f = NULL;
    if (cuModuleGetFunction(&f, m, name)) fprintf(stderr, "  missing kernel %s\n", name);
    return f;
}

static CUmodule M_basic, M_par, M_glob, M_prec;

/* ------------------------------------------------------------------ basic */

static int t_vecadd(void) {
    CUfunction f = fn(M_basic, "vecadd");
    int n = 1000003;
    size_t b = n * 4;
    float *a = malloc(b), *c = malloc(b), *o = malloc(b);
    for (int i = 0; i < n; i++) { a[i] = rndf(-1e3f, 1e3f); c[i] = rndf(-1e3f, 1e3f); }
    CUdeviceptr da, dc, dout;
    CK(cuMemAlloc(&da, b)); CK(cuMemAlloc(&dc, b)); CK(cuMemAlloc(&dout, b));
    CK(cuMemcpyHtoD(da, a, b)); CK(cuMemcpyHtoD(dc, c, b));
    void *args[] = {&da, &dc, &dout, &n};
    CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(o, dout, b));
    for (int i = 0; i < n; i++) EXPECT(o[i] == a[i] + c[i], "vecadd[%d] %f != %f", i, o[i], a[i] + c[i]);
    cuMemFree(da); cuMemFree(dc); cuMemFree(dout);
    free(a); free(c); free(o);
    return 1;
}

static int t_saxpy_gridstride_and_extra(void) {
    CUfunction f = fn(M_basic, "saxpy");
    int n = 777777;
    float alpha = 2.5f;
    size_t b = n * 4;
    float *x = malloc(b), *y = malloc(b), *o = malloc(b);
    for (int i = 0; i < n; i++) { x[i] = rndf(-10, 10); y[i] = rndf(-10, 10); }
    CUdeviceptr dx, dy;
    CK(cuMemAlloc(&dx, b)); CK(cuMemAlloc(&dy, b));
    CK(cuMemcpyHtoD(dx, x, b)); CK(cuMemcpyHtoD(dy, y, b));
    /* launch with the CU_LAUNCH_PARAM_BUFFER_POINTER protocol: {int n; float a; ptr x; ptr y} */
    struct { int n; float a; CUdeviceptr x, y; } pb = {n, alpha, dx, dy};
    size_t pbs = sizeof pb;
    void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &pb, CU_LAUNCH_PARAM_BUFFER_SIZE, &pbs, CU_LAUNCH_PARAM_END};
    CK(cuLaunchKernel(f, 64, 1, 1, 128, 1, 1, 0, 0, NULL, extra)); /* far fewer threads than n: grid-stride */
    CK(cuMemcpyDtoH(o, dy, b));
    for (int i = 0; i < n; i++) {
        float ref = alpha * x[i] + y[i];
        EXPECT(fabsf(o[i] - ref) <= 1e-5f * fmaxf(1.0f, fabsf(ref)), "saxpy[%d] %f != %f", i, o[i], ref);
    }
    cuMemFree(dx); cuMemFree(dy);
    free(x); free(y); free(o);
    return 1;
}

static uint32_t prmt_ref(uint32_t a, uint32_t b, uint32_t c) {
    uint64_t x = ((uint64_t)b << 32) | a;
    uint32_t r = 0;
    for (int i = 0; i < 4; i++) {
        uint32_t s = (c >> (4 * i)) & 0xF;
        uint32_t byte = (uint32_t)(x >> (8 * (s & 7))) & 0xFF;
        if (s & 8) byte = (byte & 0x80) ? 0xFF : 0;
        r |= byte << (8 * i);
    }
    return r;
}
static int clz32(uint32_t x) { return x ? __builtin_clz(x) : 32; }
static uint32_t brev32(uint32_t x) {
    uint32_t r = 0;
    for (int i = 0; i < 32; i++) if (x & (1u << i)) r |= 1u << (31 - i);
    return r;
}

static int t_int_ops(void) {
    CUfunction f = fn(M_basic, "int_ops");
    int n = 4096;
    uint32_t *in = malloc(n * 4), *out = malloc(n * 64);
    for (int i = 0; i < n; i++) in[i] = i < 8 ? (uint32_t[]){0, 1, 0x80000000u, 0xFFFFFFFFu, 0x7FFFFFFFu, 0x00010000u, 255, 0xdeadbeef}[i] : rnd();
    CUdeviceptr din, dout;
    CK(cuMemAlloc(&din, n * 4)); CK(cuMemAlloc(&dout, n * 64));
    CK(cuMemcpyHtoD(din, in, n * 4));
    void *args[] = {&din, &dout, &n};
    CK(cuLaunchKernel(f, n / 128, 1, 1, 128, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(out, dout, n * 64));
    for (int i = 0; i < n; i++) {
        uint32_t x = in[i];
        int32_t s = (int32_t)x;
        uint32_t ref[16];
        ref[0] = (uint32_t)__builtin_popcount(x);
        ref[1] = (uint32_t)clz32(x);
        ref[2] = brev32(x);
        ref[3] = prmt_ref(x, 0x12345678u, 0x3210 + (i & 3) * 0x1111);
        ref[4] = (uint32_t)(((((uint64_t)(x ^ 0xdeadbeefu)) << 32 | x) << (i & 31)) >> 32);
        ref[5] = (uint32_t)(((uint64_t)x * 0x9e3779b9u) >> 32);
        ref[6] = (uint32_t)(s / (i - 7 == 0 ? 3 : i - 7));
        ref[7] = (uint32_t)(s % (i + 3));
        ref[8] = (uint32_t)(s >> (i & 31));
        ref[9] = x >> (i & 31);
        ref[10] = x << (i & 31);
        ref[11] = (uint32_t)(s < -s ? s : -s);
        uint32_t m = 1000000u * (uint32_t)i;
        ref[12] = x > m ? x : m;
        int32_t s24 = (int32_t)(x << 8) >> 8;
        ref[13] = (x & 0xffu) * (x >> 24) + (uint32_t)((int64_t)s24 * 3);
        ref[14] = (uint32_t)(((uint64_t)x * 0x100000001ull) >> 17);
        ref[15] = (uint32_t)__builtin_ffs((int)x);
        for (int k = 0; k < 16; k++)
            EXPECT(out[i * 16 + k] == ref[k], "int_ops x=0x%08x i=%d op %d: got 0x%08x want 0x%08x", x, i, k, out[i * 16 + k], ref[k]);
    }
    cuMemFree(din); cuMemFree(dout);
    free(in); free(out);
    return 1;
}

static int sat_f2i(float v) {
    if (isnan(v)) return 0;
    if (v >= 2147483648.0f) return 2147483647;
    if (v <= -2147483648.0f) return (-2147483647 - 1);
    return (int)v;
}
static unsigned sat_f2u(float v) {
    if (isnan(v) || v <= 0) return 0;
    if (v >= 4294967296.0f) return 0xFFFFFFFFu;
    return (unsigned)v;
}

static int t_float_ops(void) {
    CUfunction f = fn(M_basic, "float_ops");
    int n = 8192;
    float *in = malloc(n * 4), *out = malloc(n * 32);
    int *iout = malloc(n * 16);
    for (int i = 0; i < n; i++) in[i] = i < 6 ? (float[]){0.0f, -0.0f, 2.5f, -2.5f, 1e-3f, 49.999f}[i] : rndf(-50, 50);
    CUdeviceptr din, dout, diout;
    CK(cuMemAlloc(&din, n * 4)); CK(cuMemAlloc(&dout, n * 32)); CK(cuMemAlloc(&diout, n * 16));
    CK(cuMemcpyHtoD(din, in, n * 4));
    void *args[] = {&din, &dout, &diout, &n};
    CK(cuLaunchKernel(f, n / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(out, dout, n * 32));
    CK(cuMemcpyDtoH(iout, diout, n * 16));
    for (int i = 0; i < n; i++) {
        float x = in[i];
        float ref[8] = {sqrtf(fabsf(x)), fmaf(x, 1.5f, -2.25f), fminf(x, 0.5f) + fmaxf(x, -0.5f), x / 3.0f,
                        floorf(x) + ceilf(x * 0.5f), (float)(int)x, expf(x * 0.1f), 1.0f / sqrtf(fabsf(x) + 1.0f)};
        float tol[8] = {1e-7f, 1e-6f, 0, 1e-7f, 0, 0, 2e-6f, 2e-6f};
        for (int k = 0; k < 8; k++) {
            float d = fabsf(out[i * 8 + k] - ref[k]);
            EXPECT(d <= tol[k] * fmaxf(1.0f, fabsf(ref[k])), "float_ops x=%.9g op %d: got %.9g want %.9g", x, k, out[i * 8 + k], ref[k]);
        }
        int iref[4] = {(int)rintf(x * 1000.0f), sat_f2i(x * 1e9f), x > 0 ? 1 : (x < 0 ? -1 : 0), (int)(unsigned char)sat_f2u(x * 10.0f)};
        for (int k = 0; k < 4; k++) EXPECT(iout[i * 4 + k] == iref[k], "float_ops int x=%.9g op %d: got %d want %d", x, k, iout[i * 4 + k], iref[k]);
    }
    cuMemFree(din); cuMemFree(dout); cuMemFree(diout);
    free(in); free(out); free(iout);
    return 1;
}

struct Params { float scale; int offset; float bias[2]; unsigned long long mask; };

static int t_struct_param(void) {
    CUfunction f = fn(M_basic, "struct_param");
    int n = 10000;
    int *in = malloc(n * 4);
    float *out = malloc(n * 4);
    for (int i = 0; i < n; i++) in[i] = (int)(rnd() % 100000) - 50000;
    struct Params p = {0.125f, 17, {1.5f, -3.0f}, 0xFFFFull};
    CUdeviceptr din, dout;
    CK(cuMemAlloc(&din, n * 4)); CK(cuMemAlloc(&dout, n * 4));
    CK(cuMemcpyHtoD(din, in, n * 4));
    void *args[] = {&p, &din, &dout, &n};
    CK(cuLaunchKernel(f, (n + 127) / 128, 1, 1, 128, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(out, dout, n * 4));
    for (int i = 0; i < n; i++) {
        float ref = (float)((in[i] + p.offset) & (int)p.mask) * p.scale + p.bias[i & 1];
        EXPECT(out[i] == ref, "struct_param[%d] %f != %f", i, out[i], ref);
    }
    cuMemFree(din); cuMemFree(dout);
    free(in); free(out);
    return 1;
}

static int t_f64(void) {
    CUfunction f = fn(M_basic, "f64_ops");
    int n = 5000;
    double *a = malloc(n * 8), *b = malloc(n * 8), *c = malloc(n * 8);
    for (int i = 0; i < n; i++) { a[i] = rndf(-100, 100); b[i] = rndf(-100, 100); }
    CUdeviceptr da, db, dc;
    CK(cuMemAlloc(&da, n * 8)); CK(cuMemAlloc(&db, n * 8)); CK(cuMemAlloc(&dc, n * 8));
    CK(cuMemcpyHtoD(da, a, n * 8)); CK(cuMemcpyHtoD(db, b, n * 8));
    void *args[] = {&da, &db, &dc, &n};
    CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(c, dc, n * 8));
    for (int i = 0; i < n; i++) {
        double ref = a[i] * b[i] + 0.5 * a[i] - sqrt(fabs(b[i]));
        /* loose enough for METALCUDA_F64=fast (fp32 compute); exact mode is checked bit for bit by f64_exact */
        EXPECT(fabs(c[i] - ref) <= 2e-6 * fmax(1.0, fmax(fabs(ref), fabs(a[i] * b[i]))), "f64[%d] %.10g != %.10g", i, c[i], ref);
    }
    cuMemFree(da); cuMemFree(db); cuMemFree(dc);
    free(a); free(b); free(c);
    return 1;
}

static int t_grid3d(void) {
    CUfunction f = fn(M_basic, "grid2d");
    int w = 300, h = 77, d = 3;
    float *o = malloc((size_t)w * h * d * 4);
    CUdeviceptr dout;
    CK(cuMemAlloc(&dout, (size_t)w * h * d * 4));
    void *args[] = {&dout, &w, &h};
    CK(cuLaunchKernel(f, (w + 15) / 16, (h + 7) / 8, d, 16, 8, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(o, dout, (size_t)w * h * d * 4));
    for (int z = 0; z < d; z++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                float ref = (float)(x + 1000 * y + 1000000 * z);
                EXPECT(o[(z * h + y) * w + x] == ref, "grid (%d,%d,%d) = %f want %f", x, y, z, o[(z * h + y) * w + x], ref);
            }
    cuMemFree(dout);
    free(o);
    return 1;
}

static int t_small_types(void) {
    CUfunction f = fn(M_basic, "small_types");
    int n = 3000;
    unsigned char *u8 = malloc(n);
    short *s16 = malloc(n * 2);
    int *o = malloc(n * 4);
    for (int i = 0; i < n; i++) { u8[i] = (unsigned char)rnd(); s16[i] = (short)rnd(); }
    CUdeviceptr du, ds, dout;
    CK(cuMemAlloc(&du, n)); CK(cuMemAlloc(&ds, n * 2)); CK(cuMemAlloc(&dout, n * 4));
    CK(cuMemcpyHtoD(du, u8, n)); CK(cuMemcpyHtoD(ds, s16, n * 2));
    void *args[] = {&du, &ds, &dout, &n};
    CK(cuLaunchKernel(f, (n + 63) / 64, 1, 1, 64, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(o, dout, n * 4));
    for (int i = 0; i < n; i++) {
        int ref = (int)u8[i] * 3 + (int)s16[i] - (signed char)u8[i];
        EXPECT(o[i] == ref, "small_types[%d] %d != %d", i, o[i], ref);
    }
    cuMemFree(du); cuMemFree(ds); cuMemFree(dout);
    free(u8); free(s16); free(o);
    return 1;
}

/* --------------------------------------------------------------- parallel */

static int t_reduce_smem(void) {
    CUfunction f = fn(M_par, "reduce_smem");
    int n = 1 << 22;
    float *in = malloc((size_t)n * 4);
    double ref = 0;
    for (int i = 0; i < n; i++) { in[i] = (float)(rnd() % 1000) / 1000.0f; ref += in[i]; }
    CUdeviceptr din, dout;
    CK(cuMemAlloc(&din, (size_t)n * 4)); CK(cuMemAlloc(&dout, 4));
    CK(cuMemcpyHtoD(din, in, (size_t)n * 4));
    CK(cuMemsetD32(dout, 0, 1));
    int threads = 256, blocks = (n + threads * 2 - 1) / (threads * 2);
    void *args[] = {&din, &dout, &n};
    CK(cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, threads * 4, 0, args, NULL));
    float got;
    CK(cuMemcpyDtoH(&got, dout, 4));
    EXPECT(fabs(got - ref) / ref < 1e-5, "reduce_smem %f vs %f", got, ref);
    cuMemFree(din); cuMemFree(dout);
    free(in);
    return 1;
}

static int t_reduce_warp(void) {
    CUfunction f = fn(M_par, "reduce_warp");
    int n = 1000000;
    int *in = malloc(n * 4);
    long long ref = 0;
    for (int i = 0; i < n; i++) { in[i] = (int)(rnd() % 2001) - 1000; ref += in[i]; }
    CUdeviceptr din, dout;
    CK(cuMemAlloc(&din, n * 4)); CK(cuMemAlloc(&dout, 4));
    CK(cuMemcpyHtoD(din, in, n * 4));
    CK(cuMemsetD32(dout, 0, 1));
    void *args[] = {&din, &dout, &n};
    CK(cuLaunchKernel(f, (n + 511) / 512, 1, 1, 512, 1, 1, 0, 0, args, NULL));
    int got;
    CK(cuMemcpyDtoH(&got, dout, 4));
    EXPECT(got == ref, "reduce_warp %d vs %lld", got, ref);
    cuMemFree(din); cuMemFree(dout);
    free(in);
    return 1;
}

static int t_warp_vote(void) {
    CUfunction f = fn(M_par, "warp_vote");
    int n = 256;
    int *in = malloc(n * 4);
    unsigned *out = malloc(n * 24);
    for (int i = 0; i < n; i++) in[i] = (int)(rnd() % 301) - 150 + (i >= 64 && i < 96 ? 1000 : 0);
    CUdeviceptr din, dout;
    CK(cuMemAlloc(&din, n * 4)); CK(cuMemAlloc(&dout, n * 24));
    CK(cuMemcpyHtoD(din, in, n * 4));
    void *args[] = {&din, &dout};
    CK(cuLaunchKernel(f, 2, 1, 1, 128, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(out, dout, n * 24));
    for (int w = 0; w < n / 32; w++) {
        unsigned ballot = 0;
        int any = 0, all = 1;
        for (int l = 0; l < 32; l++) {
            int v = in[w * 32 + l];
            if (v > 0) ballot |= 1u << l;
            any |= v > 100;
            all &= v > -1000;
        }
        for (int l = 0; l < 32; l++) {
            int i = w * 32 + l;
            unsigned *o = out + i * 6;
            float fx = (float)in[w * 32 + (l ^ 5)] * 0.5f;
            unsigned fxb;
            memcpy(&fxb, &fx, 4);
            EXPECT(o[0] == ballot, "ballot warp %d lane %d: 0x%08x want 0x%08x", w, l, o[0], ballot);
            EXPECT(o[1] == ((unsigned)any | ((unsigned)all << 1)), "any/all warp %d", w);
            EXPECT(o[2] == (unsigned)in[w * 32 + 7], "shfl idx lane %d", l);
            EXPECT(o[3] == (unsigned)in[l >= 3 ? i - 3 : i], "shfl up lane %d: %d want %d", l, (int)o[3], in[l >= 3 ? i - 3 : i]);
            EXPECT(o[4] == fxb, "shfl xor float lane %d", l);
            EXPECT(o[5] == (unsigned)l, "laneid %d", l);
        }
    }
    cuMemFree(din); cuMemFree(dout);
    free(in); free(out);
    return 1;
}

static int t_histogram(void) {
    CUfunction f = fn(M_par, "histogram");
    int n = 3000000;
    unsigned char *data = malloc(n);
    unsigned ref[256] = {0}, got[256];
    for (int i = 0; i < n; i++) { data[i] = (unsigned char)(rnd() % 7 == 0 ? 42 : rnd()); ref[data[i]]++; }
    CUdeviceptr dd, db;
    CK(cuMemAlloc(&dd, n)); CK(cuMemAlloc(&db, 1024));
    CK(cuMemcpyHtoD(dd, data, n));
    CK(cuMemsetD8(db, 0, 1024));
    void *args[] = {&dd, &db, &n};
    CK(cuLaunchKernel(f, 120, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(got, db, 1024));
    for (int b = 0; b < 256; b++) EXPECT(got[b] == ref[b], "bin %d: %u want %u", b, got[b], ref[b]);
    cuMemFree(dd); cuMemFree(db);
    free(data);
    return 1;
}

static int t_scan(void) {
    CUfunction f = fn(M_par, "scan_block");
    int n = 1024 * 20 + 333;
    int *in = malloc(n * 4), *out = malloc(n * 4);
    for (int i = 0; i < n; i++) in[i] = (int)(rnd() % 100);
    CUdeviceptr din, dout;
    CK(cuMemAlloc(&din, n * 4)); CK(cuMemAlloc(&dout, n * 4));
    CK(cuMemcpyHtoD(din, in, n * 4));
    void *args[] = {&din, &dout, &n};
    CK(cuLaunchKernel(f, (n + 1023) / 1024, 1, 1, 1024, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(out, dout, n * 4));
    for (int blk = 0; blk * 1024 < n; blk++) {
        int acc = 0;
        for (int t = 0; t < 1024 && blk * 1024 + t < n; t++) {
            acc += in[blk * 1024 + t];
            EXPECT(out[blk * 1024 + t] == acc, "scan block %d t %d: %d want %d", blk, t, out[blk * 1024 + t], acc);
        }
    }
    cuMemFree(din); cuMemFree(dout);
    free(in); free(out);
    return 1;
}

static int t_sgemm(void) {
    CUfunction f = fn(M_par, "sgemm_tiled");
    int M = 257, N = 190, K = 333;
    float *A = malloc((size_t)M * K * 4), *B = malloc((size_t)K * N * 4), *C = malloc((size_t)M * N * 4);
    for (int i = 0; i < M * K; i++) A[i] = rndf(-1, 1);
    for (int i = 0; i < K * N; i++) B[i] = rndf(-1, 1);
    CUdeviceptr dA, dB, dC;
    CK(cuMemAlloc(&dA, (size_t)M * K * 4)); CK(cuMemAlloc(&dB, (size_t)K * N * 4)); CK(cuMemAlloc(&dC, (size_t)M * N * 4));
    CK(cuMemcpyHtoD(dA, A, (size_t)M * K * 4)); CK(cuMemcpyHtoD(dB, B, (size_t)K * N * 4));
    void *args[] = {&dA, &dB, &dC, &M, &N, &K};
    CK(cuLaunchKernel(f, (N + 15) / 16, (M + 15) / 16, 1, 16, 16, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(C, dC, (size_t)M * N * 4));
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)A[r * K + k] * B[k * N + c];
            EXPECT(fabs(C[r * N + c] - acc) < 1e-4, "C[%d][%d] %f want %f", r, c, C[r * N + c], acc);
        }
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    free(A); free(B); free(C);
    return 1;
}

static int t_atomics(void) {
    CUfunction f = fn(M_par, "atomics");
    int n = 100000;
    float *in = malloc(n * 4);
    float fmx = -1e30f;
    int imx = -2147483647, imn = 2147483647;
    unsigned long long sum = 0;
    for (int i = 0; i < n; i++) {
        in[i] = rndf(-1000, 1000);
        if (in[i] > fmx) fmx = in[i];
        int v = (int)(in[i] * 100.0f);
        if (v > imx) imx = v;
        if (v < imn) imn = v;
        sum += (unsigned long long)i * 1000003ull;
    }
    CUdeviceptr din, dfm, dimx, dimn, dinc, dsum, dex;
    CK(cuMemAlloc(&din, n * 4));
    CK(cuMemAlloc(&dfm, 4)); CK(cuMemAlloc(&dimx, 4)); CK(cuMemAlloc(&dimn, 4)); CK(cuMemAlloc(&dinc, 4)); CK(cuMemAlloc(&dsum, 8)); CK(cuMemAlloc(&dex, 4));
    CK(cuMemcpyHtoD(din, in, n * 4));
    float init_f = -1e30f;
    int init_mx = -2147483647, init_mn = 2147483647, zero = 0, neg = -1;
    unsigned long long z64 = 0;
    CK(cuMemcpyHtoD(dfm, &init_f, 4)); CK(cuMemcpyHtoD(dimx, &init_mx, 4)); CK(cuMemcpyHtoD(dimn, &init_mn, 4));
    CK(cuMemcpyHtoD(dinc, &zero, 4)); CK(cuMemcpyHtoD(dsum, &z64, 8)); CK(cuMemcpyHtoD(dex, &neg, 4));
    void *args[] = {&din, &n, &dfm, &dimx, &dimn, &dinc, &dsum, &dex};
    CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    float gf; int gmx, gmn, gex; unsigned ginc; unsigned long long gsum;
    CK(cuMemcpyDtoH(&gf, dfm, 4)); CK(cuMemcpyDtoH(&gmx, dimx, 4)); CK(cuMemcpyDtoH(&gmn, dimn, 4));
    CK(cuMemcpyDtoH(&ginc, dinc, 4)); CK(cuMemcpyDtoH(&gsum, dsum, 8)); CK(cuMemcpyDtoH(&gex, dex, 4));
    EXPECT(gf == fmx, "CAS float max %f want %f", gf, fmx);
    EXPECT(gmx == imx && gmn == imn, "atomicMax/Min %d %d want %d %d", gmx, gmn, imx, imn);
    EXPECT(ginc == (unsigned)(n % 100), "atomicInc wrap %u want %u", ginc, n % 100);
    EXPECT(gsum == sum, "64-bit atomicAdd %llu want %llu", gsum, sum);
    EXPECT(gex >= 0 && gex < n, "atomicExch result %d", gex);
    cuMemFree(din); cuMemFree(dfm); cuMemFree(dimx); cuMemFree(dimn); cuMemFree(dinc); cuMemFree(dsum); cuMemFree(dex);
    free(in);
    return 1;
}

static int t_local_array(void) {
    CUfunction f = fn(M_par, "local_array");
    int n = 4096;
    int *idx = malloc(n * 4), *out = malloc(n * 4);
    for (int i = 0; i < n; i++) idx[i] = (int)(rnd() % 1000);
    CUdeviceptr di, dout;
    CK(cuMemAlloc(&di, n * 4)); CK(cuMemAlloc(&dout, n * 4));
    CK(cuMemcpyHtoD(di, idx, n * 4));
    void *args[] = {&di, &dout, &n};
    CK(cuLaunchKernel(f, n / 128, 1, 1, 128, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(out, dout, n * 4));
    for (int i = 0; i < n; i++) {
        int acc = 0;
        for (int k = 0; k < 8; k++) { int j = (idx[i] + k * 5) % 24; acc += j * j + i; }
        EXPECT(out[i] == acc, "local_array[%d] %d want %d", i, out[i], acc);
    }
    cuMemFree(di); cuMemFree(dout);
    free(idx); free(out);
    return 1;
}

static int t_sync_count(void) {
    CUfunction f = fn(M_par, "sync_count");
    int blocks = 8, threads = 192, n = blocks * threads;
    int *in = malloc(n * 4), out[24];
    for (int i = 0; i < n; i++) in[i] = (int)(rnd() % 1000) - (i / threads == 3 ? 500 : 0);
    in[5 * threads + 17] = 42;
    CUdeviceptr din, dout;
    CK(cuMemAlloc(&din, n * 4)); CK(cuMemAlloc(&dout, sizeof out));
    CK(cuMemcpyHtoD(din, in, n * 4));
    void *args[] = {&din, &dout};
    CK(cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(out, dout, sizeof out));
    for (int b = 0; b < blocks; b++) {
        int c = 0, a = 1, o = 0;
        for (int t = 0; t < threads; t++) { int v = in[b * threads + t]; c += v & 1; a &= v >= 0; o |= v == 42; }
        EXPECT(out[b * 3] == c && (out[b * 3 + 1] != 0) == a && (out[b * 3 + 2] != 0) == o, "block %d: count %d/%d and %d/%d or %d/%d", b, out[b * 3], c,
               out[b * 3 + 1], a, out[b * 3 + 2], o);
    }
    cuMemFree(din); cuMemFree(dout);
    free(in);
    return 1;
}


static int t_loop_then_barrier(void) {
    CUfunction f = fn(M_par, "loop_then_barrier");
    int blocks = 4, threads = 256, n = blocks * threads;
    int *out = malloc(n * 4);
    CUdeviceptr d;
    CK(cuMemAlloc(&d, n * 4));
    void *args[] = {&d};
    CK(cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(out, d, n * 4));
    int acc[256];
    for (int t = 0; t < threads; t++) {
        int i = 0, a = 0;
        do { a = (int)((unsigned)a * 1664525u + 1013904223u + (unsigned)i); i++; } while (i < (t % 37) + 1);
        acc[t] = a;
    }
    for (int b = 0; b < blocks; b++)
        for (int t = 0; t < threads; t++) {
            int ref = acc[(t + 1) % threads] * 1000 + acc[t ^ 1];
            EXPECT(out[b * threads + t] == ref, "block %d t %d: %d want %d", b, t, out[b * threads + t], ref);
        }
    cuMemFree(d);
    free(out);
    return 1;
}

/* Hand-written PTX whose loop has two entries (irreducible CFG): the
 * translator must fall back to its block-dispatch loop and still be correct. */
static const char *IRREDUCIBLE_PTX =
    ".version 7.0\n.target sm_80\n.address_size 64\n"
    ".visible .entry irreducible(.param .u64 out)\n{\n"
    "  .reg .pred %p<4>;\n  .reg .b32 %r<8>;\n  .reg .b64 %rd<4>;\n"
    "  mov.u32 %r1, %tid.x;\n  mov.u32 %r2, 0;\n"
    "  and.b32 %r3, %r1, 1;\n  setp.ne.u32 %p1, %r3, 0;\n"
    "  @%p1 bra LB;\n"
    "LA:\n  add.u32 %r2, %r2, 1;\n  setp.lt.u32 %p2, %r2, %r1;\n  @%p2 bra LB;\n  bra LEND;\n"
    "LB:\n  add.u32 %r2, %r2, 3;\n  setp.lt.u32 %p3, %r2, %r1;\n  @%p3 bra LA;\n"
    "LEND:\n  ld.param.u64 %rd1, [out];\n  mul.wide.u32 %rd2, %r1, 4;\n  add.s64 %rd3, %rd1, %rd2;\n"
    "  st.global.u32 [%rd3], %r2;\n  ret;\n}\n";

static int t_irreducible(void) {
    CUmodule m;
    CUfunction f;
    CK(cuModuleLoadData(&m, IRREDUCIBLE_PTX));
    CK(cuModuleGetFunction(&f, m, "irreducible"));
    int n = 128;
    unsigned out[128];
    CUdeviceptr d;
    CK(cuMemAlloc(&d, sizeof out));
    void *args[] = {&d};
    CK(cuLaunchKernel(f, 1, 1, 1, n, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(out, d, sizeof out));
    for (unsigned t = 0; t < (unsigned)n; t++) {
        unsigned x = 0;
        int at_b = (t & 1) != 0;
        for (;;) {
            if (!at_b) { x += 1; if (x < t) { at_b = 1; continue; } break; }
            x += 3;
            if (x < t) { at_b = 0; continue; }
            break;
        }
        EXPECT(out[t] == x, "irreducible t=%u: %u want %u", t, out[t], x);
    }
    cuMemFree(d);
    cuModuleUnload(m);
    return 1;
}



/* Async copies on a stream: stream order (HtoD -> kernel -> DtoH), the host
 * source may be reused/freed as soon as cuMemcpyHtoDAsync returns, and the
 * reported capabilities match the implementation (no concurrency claimed). */
static int t_async_semantics(void) {
    CUfunction f = fn(M_basic, "saxpy");
    int n = 1 << 20;
    float alpha = 2.0f;
    CUstream s;
    CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    CUdeviceptr dx, dy;
    CK(cuMemAlloc(&dx, (size_t)n * 4)); CK(cuMemAlloc(&dy, (size_t)n * 4));
    float *out = malloc((size_t)n * 4);
    for (int round = 0; round < 3; round++) {
        float *hx = malloc((size_t)n * 4), *hy = malloc((size_t)n * 4);
        for (int i = 0; i < n; i++) { hx[i] = (float)(i % 1000) + round; hy[i] = 1.0f; }
        CK(cuMemcpyHtoDAsync(dx, hx, (size_t)n * 4, s));
        CK(cuMemcpyHtoDAsync(dy, hy, (size_t)n * 4, s));
        memset(hx, 0xFF, (size_t)n * 4); /* reuse the source immediately ... */
        free(hx);                         /* ... and free it */
        free(hy);
        void *args[] = {&n, &alpha, &dx, &dy};
        CK(cuLaunchKernel(f, 256, 1, 1, 256, 1, 1, 0, s, args, NULL));
        CK(cuMemcpyDtoHAsync(out, dy, (size_t)n * 4, s)); /* must observe the kernel's writes */
        CK(cuStreamSynchronize(s));
        for (int i = 0; i < n; i++)
            EXPECT(out[i] == alpha * ((float)(i % 1000) + round) + 1.0f, "round %d [%d]: %f", round, i, out[i]);
    }
    int conc = -1, overlap = -1, engines = -1;
    CK(cuDeviceGetAttribute(&conc, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, 0));
    CK(cuDeviceGetAttribute(&overlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, 0));
    CK(cuDeviceGetAttribute(&engines, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, 0));
    EXPECT(conc == 0 && overlap == 0 && engines == 0, "capabilities must not claim concurrency (%d %d %d)", conc, overlap, engines);
    cuMemFree(dx); cuMemFree(dy); cuStreamDestroy(s);
    free(out);
    return 1;
}

/* ------------------------------------------------------ 64-bit atomics */

static int cmp_u64(const void *a, const void *b) {
    unsigned long long x = *(const unsigned long long *)a, y = *(const unsigned long long *)b;
    return x < y ? -1 : x > y;
}

static int t_atom64_unique(void) {
    CUfunction f = fn(M_prec, "atom64_add_unique");
    /* 16k threads = 512 SIMD-groups on one address. Far higher contention can hang:
     * Metal has no forward-progress guarantee for cross-threadgroup spinning (README). */
    int n = 16384;
    unsigned long long incs[2] = {1ull, 0x100000001ull};
    unsigned long long *olds = malloc((size_t)n * 8);
    CUdeviceptr dc, dol;
    CK(cuMemAlloc(&dc, 8)); CK(cuMemAlloc(&dol, (size_t)n * 8));
    for (int t = 0; t < 2; t++) {
        unsigned long long start = 0xFFFFFF00ull, inc = incs[t], fin;
        CK(cuMemcpyHtoD(dc, &start, 8));
        void *args[] = {&dc, &dol, &inc, &n};
        CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
        CK(cuMemcpyDtoH(olds, dol, (size_t)n * 8));
        CK(cuMemcpyDtoH(&fin, dc, 8));
        EXPECT(fin == start + (unsigned long long)n * inc, "final counter 0x%llx want 0x%llx", fin, start + (unsigned long long)n * inc);
        qsort(olds, (size_t)n, 8, cmp_u64);
        for (int i = 0; i < n; i++)
            EXPECT(olds[i] == start + (unsigned long long)i * inc, "inc 0x%llx: returned old values are not a permutation (rank %d: 0x%llx want 0x%llx)",
                   inc, i, olds[i], start + (unsigned long long)i * inc);
    }
    cuMemFree(dc); cuMemFree(dol);
    free(olds);
    return 1;
}

static int t_atom64_misc(void) {
    CUfunction f = fn(M_prec, "atom64_misc");
    /* every thread runs a 64-bit CAS retry loop on ONE address: O(n^2) lock-protected
     * attempts, so keep n moderate (see README: 64-bit atomics) */
    int n = 4096;
    const unsigned long long K = 0xF0F0F0F0F0F0ull;
    unsigned long long init[5] = {0xFFFFFFFFull, 0x42ull, 0x1ull, 0xFFFFFFFFFFFFFFFFull, 0};
    long long sinit = (long long)0x8000000000000000ull;
    CUdeviceptr d[5], dxo, dmo, dsm;
    for (int i = 0; i < 4; i++) { CK(cuMemAlloc(&d[i], 8)); CK(cuMemcpyHtoD(d[i], &init[i], 8)); }
    CK(cuMemAlloc(&dsm, 8)); CK(cuMemcpyHtoD(dsm, &sinit, 8));
    CK(cuMemAlloc(&dxo, (size_t)n * 8)); CK(cuMemAlloc(&dmo, (size_t)n * 8));
    void *args[] = {&d[0], &d[1], &dxo, &d[2], &d[3], &dmo, &dsm, &n};
    CK(cuLaunchKernel(f, (n + 127) / 128, 1, 1, 128, 1, 1, 0, 0, args, NULL));
    unsigned long long fin[4];
    long long sfin;
    for (int i = 0; i < 4; i++) CK(cuMemcpyDtoH(&fin[i], d[i], 8));
    CK(cuMemcpyDtoH(&sfin, dsm, 8));
    EXPECT(fin[0] == init[0] + (unsigned long long)n * 0x100000001ull, "64-bit CAS loop counter 0x%llx", fin[0]);
    /* exchange conservation: returned values + final == initial + all written values */
    unsigned long long *got = malloc((size_t)(n + 1) * 8), *want = malloc((size_t)(n + 1) * 8), *mo = malloc((size_t)n * 8);
    CK(cuMemcpyDtoH(got, dxo, (size_t)n * 8));
    CK(cuMemcpyDtoH(mo, dmo, (size_t)n * 8));
    got[n] = fin[1];
    want[n] = init[1];
    unsigned long long mx = init[2], mn = init[3];
    long long smx = sinit;
    for (int i = 0; i < n; i++) {
        unsigned long long v = 0x123456789ull * (unsigned long long)(i + 1);
        want[i] = v;
        if ((v ^ K) > mx) mx = v ^ K;
        if ((v ^ K) < mn) mn = v ^ K;
        if ((long long)(v ^ 0x8000000000000000ull) > smx) smx = (long long)(v ^ 0x8000000000000000ull);
    }
    qsort(got, (size_t)n + 1, 8, cmp_u64);
    qsort(want, (size_t)n + 1, 8, cmp_u64);
    for (int i = 0; i <= n; i++) EXPECT(got[i] == want[i], "64-bit exchange lost or duplicated a value (rank %d)", i);
    EXPECT(fin[2] == mx && fin[3] == mn && sfin == smx, "64-bit min/max 0x%llx 0x%llx %lld", fin[2], fin[3], sfin);
    for (int i = 0; i < n; i++) EXPECT(mo[i] <= mx, "atomicMax returned a value above the final maximum");
    for (int i = 0; i < 4; i++) cuMemFree(d[i]);
    cuMemFree(dsm); cuMemFree(dxo); cuMemFree(dmo);
    free(got); free(want); free(mo);
    return 1;
}

static int t_atom64_shared(void) {
    CUfunction f = fn(M_prec, "atom64_shared");
    int blocks = 16, threads = 512, n = blocks * threads;
    unsigned long long *olds = malloc((size_t)n * 8), tot[16];
    CUdeviceptr dol, dt;
    CK(cuMemAlloc(&dol, (size_t)n * 8)); CK(cuMemAlloc(&dt, sizeof tot));
    void *args[] = {&dol, &dt};
    CK(cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(olds, dol, (size_t)n * 8));
    CK(cuMemcpyDtoH(tot, dt, sizeof tot));
    for (int b = 0; b < blocks; b++) {
        EXPECT(tot[b] == 0xFFFFFFF0ull + (unsigned long long)threads, "shared 64-bit counter block %d: 0x%llx", b, tot[b]);
        qsort(olds + b * threads, (size_t)threads, 8, cmp_u64);
        for (int t = 0; t < threads; t++) EXPECT(olds[b * threads + t] == 0xFFFFFFF0ull + (unsigned long long)t, "shared 64-bit old values block %d", b);
    }
    cuMemFree(dol); cuMemFree(dt);
    free(olds);
    return 1;
}

static int t_atom_f64(void) {
    CUfunction f = fn(M_prec, "atom_f64_add");
    int n = 100000;
    double *v = malloc((size_t)n * 8), ref = 0, sum = 0;
    for (int i = 0; i < n; i++) { v[i] = (double)(int)(rnd() % 2001 - 1000) / 1024.0 + (double)(rnd() % 1000) * 4096.0; ref += v[i]; }
    CUdeviceptr dv, ds;
    CK(cuMemAlloc(&dv, (size_t)n * 8)); CK(cuMemAlloc(&ds, 8));
    CK(cuMemcpyHtoD(dv, v, (size_t)n * 8)); CK(cuMemcpyHtoD(ds, &sum, 8));
    void *args[] = {&dv, &ds, &n};
    CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(&sum, ds, 8));
    EXPECT(sum == ref, "double atomicAdd %.17g want %.17g (all partial sums are exactly representable)", sum, ref);
    cuMemFree(dv); cuMemFree(ds);
    free(v);
    return 1;
}

/* ------------------------------------------------------ IEEE binary64 */

static unsigned long long dbits(double d) { unsigned long long u; memcpy(&u, &d, 8); return u; }
static double bitsd(unsigned long long u) { double d; memcpy(&d, &u, 8); return d; }
static unsigned long long rnd64(void) { return ((unsigned long long)rnd() << 32) | rnd(); }

static double gen_double(int i) {
    static const double special[] = {0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, -2.5, 3.0, 1e308, -1e308, 4.9e-324, 2.2250738585072014e-308,
                                     2.225073858507201e-308, 9007199254740993.0, 1.7976931348623157e308, 0.1, 1e-300, 1e300, 123456789.987654321};
    int nsp = (int)(sizeof special / sizeof special[0]);
    switch (i % 5) {
    case 0: return special[rnd() % nsp] * ((rnd() & 1) ? 1.0 : -1.0);
    case 1: return bitsd(rnd64());                                                     /* any class incl. NaN/inf/subnormal */
    case 2: return bitsd((rnd64() & 0x800FFFFFFFFFFFFFull) | (0x3FEull << 52) | ((unsigned long long)(rnd() % 4) << 52)); /* similar magnitudes: cancellation */
    case 3: return bitsd((rnd64() & 0x800FFFFFFFFFFFFFull) | ((unsigned long long)(rnd() % 120) << 52));                   /* near underflow */
    default: return ldexp((double)(int)(rnd() % 2000001 - 1000000) + (double)rnd() / 4294967296.0, (int)(rnd() % 80) - 40);
    }
}

static int same_bits(unsigned long long got, double want, int zero_sign_free) {
    if (want != want) return (got & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull;
    if (zero_sign_free && want == 0.0) return (got & 0x7FFFFFFFFFFFFFFFull) == 0;
    return got == dbits(want);
}
/* PTX min/max.f64: a NaN operand (quiet or signalling) yields the other operand */
static double ptx_fmin(double x, double y) { return x != x ? y : (y != y ? x : fmin(x, y)); }
static double ptx_fmax(double x, double y) { return x != x ? y : (y != y ? x : fmax(x, y)); }
static int sat_d2i_rn(double y) {
    if (y != y) return 0;
    double r = rint(y);
    if (r >= 2147483647.0) return 2147483647;
    if (r <= -2147483648.0) return (-2147483647 - 1);
    return (int)r;
}

static int t_f64_exact(void) {
    CUfunction f = fn(M_prec, "f64_exact");
    int n = 1 << 18;
    double *a = malloc((size_t)n * 8), *b = malloc((size_t)n * 8), *c = malloc((size_t)n * 8);
    unsigned long long *o = malloc((size_t)n * 12 * 8);
    int *io = malloc((size_t)n * 16);
    float *fo = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) {
        a[i] = gen_double(i);
        b[i] = (i % 7 == 2) ? a[i] * (1.0 + ldexp(1.0, -(int)(rnd() % 53))) : gen_double(i / 3 + (int)(rnd() % 5));
        c[i] = (i % 11 == 3) ? -(a[i] * b[i]) : gen_double(i + 1);
    }
    CUdeviceptr da, db, dc, dout, dio, dfo;
    CK(cuMemAlloc(&da, (size_t)n * 8)); CK(cuMemAlloc(&db, (size_t)n * 8)); CK(cuMemAlloc(&dc, (size_t)n * 8));
    CK(cuMemAlloc(&dout, (size_t)n * 96)); CK(cuMemAlloc(&dio, (size_t)n * 16)); CK(cuMemAlloc(&dfo, (size_t)n * 4));
    CK(cuMemcpyHtoD(da, a, (size_t)n * 8)); CK(cuMemcpyHtoD(db, b, (size_t)n * 8)); CK(cuMemcpyHtoD(dc, c, (size_t)n * 8));
    void *args[] = {&da, &db, &dc, &dout, &dio, &dfo, &n};
    CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(o, dout, (size_t)n * 96)); CK(cuMemcpyDtoH(io, dio, (size_t)n * 16)); CK(cuMemcpyDtoH(fo, dfo, (size_t)n * 4));
    static const char *names[12] = {"add", "sub", "mul", "div", "sqrt", "fma", "min", "max", "i64->f64*0.1", "floor+ceil", "f64->f32->f64", "rint"};
    long bad[12] = {0};
    long ibad = 0;
    for (int i = 0; i < n; i++) {
        double x = a[i], y = b[i], z = c[i];
        double ref[12] = {x + y, x - y, x * y, x / y, sqrt(x), fma(x, y, z), ptx_fmin(x, y), ptx_fmax(x, y), (double)(long long)i * 0.1,
                          floor(x) + ceil(y), (double)(float)z, rint(x)};
        for (int k = 0; k < 12; k++)
            if (!same_bits(o[12 * i + k], ref[k], k == 6 || k == 7)) {
                if (bad[k]++ < 3)
                    fprintf(stderr, "  f64 %s(%a, %a, %a): got %a want %a\n", names[k], x, y, z, bitsd(o[12 * i + k]), ref[k]);
            }
        int iref[4] = {(x < y) | ((x == y) << 1) | ((x > y) << 2) | ((x != x) << 3), (int)ptx_fmin(ptx_fmax(x, -2e9), 2e9), sat_d2i_rn(y),
                       (int)(long long)ptx_fmin(ptx_fmax(z, -9e18), 9e18)};
        for (int k = 0; k < 4; k++)
            if (io[4 * i + k] != iref[k] && ibad++ < 3) fprintf(stderr, "  f64 int op %d (%a, %a, %a): got %d want %d\n", k, x, y, z, io[4 * i + k], iref[k]);
        float fr = (float)x;
        if (memcmp(&fr, &fo[i], 4) && !(fr != fr && fo[i] != fo[i]) && ibad++ < 3) fprintf(stderr, "  f64->f32 %a: got %a want %a\n", x, fo[i], fr);
    }
    long total = ibad;
    for (int k = 0; k < 12; k++) total += bad[k];
    cuMemFree(da); cuMemFree(db); cuMemFree(dc); cuMemFree(dout); cuMemFree(dio); cuMemFree(dfo);
    free(a); free(b); free(c); free(o); free(io); free(fo);
    for (int k = 0; k < 12; k++) if (bad[k]) fprintf(stderr, "  %s: %ld mismatches\n", names[k], bad[k]);
    if (ibad) fprintf(stderr, "  integer/f32 conversions & compares: %ld mismatches\n", ibad);
    EXPECT(total == 0, "%ld of %d results differ from IEEE binary64", total, n * 17);
    return 1;
}

/* ---------------------------------------------------------------- globals */

static int t_globals(void) {
    CUfunction f = fn(M_glob, "poly_eval");
    int n = 1000;
    float x[1000], y[1000];
    for (int i = 0; i < n; i++) x[i] = rndf(-2, 2);
    CUdeviceptr dx, dy, dcounter, dcoeffs;
    size_t sz;
    CK(cuMemAlloc(&dx, sizeof x)); CK(cuMemAlloc(&dy, sizeof y));
    CK(cuMemcpyHtoD(dx, x, sizeof x));
    CK(cuModuleGetGlobal(&dcounter, &sz, M_glob, "counter"));
    EXPECT(sz == 4, "counter size %zu", sz);
    CK(cuModuleGetGlobal(&dcoeffs, &sz, M_glob, "coeffs"));
    EXPECT(sz == 16, "coeffs size %zu", sz);
    float c[4];
    CK(cuMemcpyDtoH(c, dcoeffs, 16));
    EXPECT(c[0] == 0.5f && c[1] == -1.25f && c[2] == 2.0f && c[3] == 3.75f, "__constant__ initializer %f %f %f %f", c[0], c[1], c[2], c[3]);
    unsigned lut[8] = {1, 1, 2, 3, 5, 8, 13, 21};
    for (int round = 0; round < 2; round++) {
        void *args[] = {&dx, &dy, &n};
        CK(cuLaunchKernel(f, (n + 127) / 128, 1, 1, 128, 1, 1, 0, 0, args, NULL));
        CK(cuMemcpyDtoH(y, dy, sizeof y));
        for (int i = 0; i < n; i++) {
            float v = x[i];
            float ref = ((c[3] * v + c[2]) * v + c[1]) * v + c[0] + (float)lut[i & 7];
            EXPECT(fabsf(y[i] - ref) < 1e-4f, "poly round %d [%d] %f want %f", round, i, y[i], ref);
        }
        /* update the __constant__ table from the host, as cudaMemcpyToSymbol does */
        c[0] = 10.0f; c[3] = -1.0f;
        CK(cuMemcpyHtoD(dcoeffs, c, 16));
    }
    int cnt;
    CK(cuMemcpyDtoH(&cnt, dcounter, 4));
    EXPECT(cnt == 2 * n, "device global counter %d want %d", cnt, 2 * n);
    cuMemFree(dx); cuMemFree(dy);
    return 1;
}

/* ------------------------------------------------------- memory & runtime */

static int t_memory_kinds(void) {
    CUfunction f = fn(M_basic, "vecadd");
    int n = 50000;
    size_t b = n * 4;
    /* managed memory: host writes directly, kernel receives the same pointer */
    CUdeviceptr ma, mb, mc;
    CK(cuMemAllocManaged(&ma, b, CU_MEM_ATTACH_GLOBAL));
    CK(cuMemAllocManaged(&mb, b, CU_MEM_ATTACH_GLOBAL));
    CK(cuMemAllocManaged(&mc, b, CU_MEM_ATTACH_GLOBAL));
    float *a = (float *)(uintptr_t)ma, *bb = (float *)(uintptr_t)mb, *c = (float *)(uintptr_t)mc;
    for (int i = 0; i < n; i++) { a[i] = (float)i; bb[i] = 2.0f * i; }
    void *args[] = {&ma, &mb, &mc, &n};
    CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuCtxSynchronize());
    for (int i = 0; i < n; i++) EXPECT(c[i] == 3.0f * i, "managed[%d] %f", i, c[i]);
    unsigned is_managed = 0;
    CK(cuPointerGetAttribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, ma));
    EXPECT(is_managed == 1, "managed attribute");

    /* pinned host memory + device pointer alias */
    float *h;
    CK(cuMemHostAlloc((void **)&h, b, CU_MEMHOSTALLOC_DEVICEMAP));
    CUdeviceptr hd;
    CK(cuMemHostGetDevicePointer(&hd, h, 0));
    void *args2[] = {&ma, &mb, &hd, &n};
    CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args2, NULL));
    CK(cuCtxSynchronize());
    for (int i = 0; i < n; i++) EXPECT(h[i] == 3.0f * i, "pinned[%d] %f", i, h[i]);

    /* registered (malloc'ed, unaligned) host memory */
    float *reg = (float *)malloc(b + 12) + 3;
    for (int i = 0; i < n; i++) reg[i] = -1.0f;
    CK(cuMemHostRegister(reg, b, 0));
    CUdeviceptr rd;
    CK(cuMemHostGetDevicePointer(&rd, reg, 0));
    void *args3[] = {&ma, &ma, &rd, &n};
    CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args3, NULL));
    CK(cuCtxSynchronize());
    for (int i = 0; i < n; i++) EXPECT(reg[i] == 2.0f * i, "registered[%d] %f", i, reg[i]);
    CK(cuMemHostUnregister(reg));
    free(reg - 3);

    /* device-to-device copies, memsets, address ranges */
    CUdeviceptr d1, d2;
    CK(cuMemAlloc(&d1, b)); CK(cuMemAlloc(&d2, b));
    CK(cuMemsetD32(d1, 0x3f800000u, n)); /* 1.0f */
    CK(cuMemcpyDtoD(d2, d1, b));
    float probe[4];
    CK(cuMemcpyDtoH(probe, d2 + 400, 16));
    EXPECT(probe[0] == 1.0f && probe[3] == 1.0f, "memset/DtoD %f", probe[0]);
    CUdeviceptr base; size_t rs;
    CK(cuMemGetAddressRange(&base, &rs, d2 + 1234));
    EXPECT(base == d2 && rs == b, "address range");
    EXPECT(cuMemcpyHtoD(0x1234, probe, 4) == CUDA_ERROR_INVALID_VALUE, "bad device pointer must fail");

    cuMemFree(d1); cuMemFree(d2);
    cuMemFree(ma); cuMemFree(mb); cuMemFree(mc);
    cuMemFreeHost(h);
    return 1;
}

static int t_vmm(void) {
    CUfunction f = fn(M_basic, "vecadd");
    size_t gran;
    CUmemAllocationProp prop = {0};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    CK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t size = 4 * gran;
    CUdeviceptr va;
    CK(cuMemAddressReserve(&va, size, 0, 0, 0));
    CUmemGenericAllocationHandle h;
    CK(cuMemCreate(&h, size, &prop, 0));
    CK(cuMemMap(va, size, 0, h, 0));
    CUmemAccessDesc acc = {{CU_MEM_LOCATION_TYPE_DEVICE, 0}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    CK(cuMemSetAccess(va, size, &acc, 1));
    int n = (int)(size / 12);
    float *x = malloc(n * 4);
    for (int i = 0; i < n; i++) x[i] = (float)(i % 1000);
    CUdeviceptr a = va, b = va + (size_t)n * 4, c = va + (size_t)n * 8;
    CK(cuMemcpyHtoD(a, x, n * 4)); CK(cuMemcpyHtoD(b, x, n * 4));
    void *args[] = {&a, &b, &c, &n};
    CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuMemcpyDtoH(x, c, n * 4));
    for (int i = 0; i < n; i++) EXPECT(x[i] == 2.0f * (i % 1000), "vmm[%d] %f", i, x[i]);
    CK(cuMemUnmap(va, size));
    CK(cuMemRelease(h));
    CK(cuMemAddressFree(va, size));
    free(x);
    return 1;
}

static int t_events_streams(void) {
    CUfunction f = fn(M_par, "sgemm_tiled");
    int M = 512, N = 512, K = 512;
    CUdeviceptr dA, dB, dC;
    CK(cuMemAlloc(&dA, (size_t)M * K * 4)); CK(cuMemAlloc(&dB, (size_t)K * N * 4)); CK(cuMemAlloc(&dC, (size_t)M * N * 4));
    CK(cuMemsetD32(dA, 0x3f800000u, M * K)); CK(cuMemsetD32(dB, 0x40000000u, K * N));
    CUstream s;
    CUevent e0, e1;
    CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    CK(cuEventCreate(&e0, 0)); CK(cuEventCreate(&e1, 0));
    void *args[] = {&dA, &dB, &dC, &M, &N, &K};
    CK(cuEventRecord(e0, s));
    int iters = 20;
    for (int it = 0; it < iters; it++) CK(cuLaunchKernel(f, N / 16, M / 16, 1, 16, 16, 1, 0, s, args, NULL));
    CK(cuEventRecord(e1, s));
    CK(cuEventSynchronize(e1));
    EXPECT(cuEventQuery(e1) == CUDA_SUCCESS && cuStreamQuery(s) == CUDA_SUCCESS, "event/stream query after sync");
    float ms = 0;
    CK(cuEventElapsedTime(&ms, e0, e1));
    float probe;
    CK(cuMemcpyDtoH(&probe, dC + 4 * 1000, 4));
    EXPECT(probe == 2.0f * K, "sgemm value %f", probe);
    double gflops = 2.0 * M * N * K * iters / (ms * 1e-3) / 1e9;
    printf("    sgemm_tiled 512^3 x%d: %.3f ms total (%.1f GFLOP/s, naive tiled kernel)\n", iters, ms, gflops);
    EXPECT(ms > 0 && ms < 10000, "elapsed time %f", ms);
    cuEventDestroy(e0); cuEventDestroy(e1); cuStreamDestroy(s);
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    return 1;
}

static int t_errors_and_queries(void) {
    CUmodule m;
    /* invalid PTX */
    EXPECT(cuModuleLoadData(&m, ".version 7.0\n.target sm_80\n.address_size 64\n.visible .entry k() { foo.bar %r1; }") == CUDA_ERROR_INVALID_PTX,
           "bad PTX must fail with INVALID_PTX");
    /* SASS-only image */
    unsigned char elf[64] = {0x7f, 'E', 'L', 'F'};
    EXPECT(cuModuleLoadData(&m, elf) == CUDA_ERROR_NO_BINARY_FOR_GPU, "ELF cubin must be rejected");
    /* launch configuration limits */
    CUfunction f = fn(M_basic, "vecadd");
    CUdeviceptr d;
    CK(cuMemAlloc(&d, 64));
    int n = 0;
    void *args[] = {&d, &d, &d, &n};
    EXPECT(cuLaunchKernel(f, 1, 1, 1, 2048, 1, 1, 0, 0, args, NULL) == CUDA_ERROR_INVALID_VALUE, "2048 threads/block must fail");
    EXPECT(cuLaunchKernel(f, 0, 1, 1, 32, 1, 1, 0, 0, args, NULL) == CUDA_ERROR_INVALID_VALUE, "empty grid must fail");
    CUfunction r = fn(M_par, "reduce_smem");
    void *rargs[] = {&d, &d, &n};
    EXPECT(cuLaunchKernel(r, 1, 1, 1, 32, 1, 1, 64 * 1024, 0, rargs, NULL) == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, "64 KiB shared must fail");
    cuMemFree(d);
    /* entry point lookup as used by libcudart >= 11.3 */
    void *p = NULL;
    CUdriverProcAddressQueryResult st;
    CK(cuGetProcAddress("cuMemAlloc", &p, 12040, 0, &st));
    EXPECT(p != NULL && st == CU_GET_PROC_ADDRESS_SUCCESS, "cuGetProcAddress(cuMemAlloc)");
    CUdeviceptr q = 0;
    CK(((CUresult(*)(CUdeviceptr *, size_t))p)(&q, 256));
    EXPECT(q != 0, "allocation through proc address");
    cuMemFree(q);
    EXPECT(cuGetProcAddress("cuNoSuchFunction", &p, 12040, 0, &st) == CUDA_ERROR_NOT_FOUND, "unknown symbol");
    /* function attributes */
    int mt = 0, ss = 0;
    CK(cuFuncGetAttribute(&mt, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn(M_par, "sgemm_tiled")));
    CK(cuFuncGetAttribute(&ss, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn(M_par, "sgemm_tiled")));
    EXPECT(mt >= 256 && ss == 2048, "func attributes %d %d", mt, ss);
    return 1;
}

static int t_many_launches(void) {
    /* 10k back-to-back dependent launches: exercises batching + ordering */
    CUfunction f = fn(M_basic, "saxpy");
    int n = 4096;
    float one = 1.0f;
    CUdeviceptr dx, dy;
    CK(cuMemAlloc(&dx, n * 4)); CK(cuMemAlloc(&dy, n * 4));
    CK(cuMemsetD32(dx, 0x3f800000u, n));
    CK(cuMemsetD32(dy, 0, n));
    void *args[] = {&n, &one, &dx, &dy};
    int iters = 10000;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) CK(cuLaunchKernel(f, 16, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuCtxSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1e3 / iters;
    float y[4];
    CK(cuMemcpyDtoH(y, dy + 4 * 100, 16));
    EXPECT(y[0] == (float)iters, "dependent launches %f want %d", y[0], iters);
    printf("    %d dependent launches: %.2f us/launch end-to-end\n", iters, us);
    cuMemFree(dx); cuMemFree(dy);
    return 1;
}

typedef struct { const char *name; int (*fn)(void); } test_t;

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1) g_dir = argv[1];
    const char *only = argc > 2 ? argv[2] : NULL; /* optional: run tests whose name contains this */
    if (cuInit(0)) { fprintf(stderr, "cuInit failed\n"); return 1; }
    CUdevice dev;
    CUcontext ctx;
    cuDeviceGet(&dev, 0);
    cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);
    char name[256];
    cuDeviceGetName(name, sizeof name, dev);
    printf("MetaCuda driver suite on %s\n", name);
    M_basic = load("basic");
    M_par = load("parallel");
    M_glob = load("globals");
    M_prec = load("precision");
    if (!M_basic || !M_par || !M_glob || !M_prec) return 1;
    test_t tests[] = {
        {"vecadd (1M elems, odd size)", t_vecadd},
        {"saxpy grid-stride + extra param buffer", t_saxpy_gridstride_and_extra},
        {"integer & bit ops (popc/clz/brev/prmt/shf/mulhi/div/mul24)", t_int_ops},
        {"float ops & conversions (rounding, saturation)", t_float_ops},
        {"by-value struct parameter", t_struct_param},
        {"f64 kernel (a*b + 0.5a - sqrt|b|)", t_f64},
        {"3-D grid / 2-D blocks", t_grid3d},
        {"8/16-bit loads, sign extension", t_small_types},
        {"shared-memory tree reduction + float atomicAdd", t_reduce_smem},
        {"warp shuffle reduction (__shfl_down/xor_sync)", t_reduce_warp},
        {"warp vote/ballot/shfl idx/up/xor", t_warp_vote},
        {"shared-memory histogram (shared + global atomics)", t_histogram},
        {"block scan (barriers in loops, 1024 threads)", t_scan},
        {"tiled SGEMM (static shared memory, 2-D)", t_sgemm},
        {"atomics: CAS loop, min/max, inc wrap, exch, 64-bit add", t_atomics},
        {"local memory (dynamically indexed per-thread array)", t_local_array},
        {"__syncthreads_count/and/or (bar.red)", t_sync_count},
        {"divergent loop then barrier + shuffle (loop-exit placement)", t_loop_then_barrier},
        {"irreducible CFG (hand-written PTX, dispatch-loop fallback)", t_irreducible},
        {"64-bit atomicAdd: returned old values are a permutation", t_atom64_unique},
        {"64-bit CAS loop, exchange conservation, min/max", t_atom64_misc},
        {"64-bit atomics in shared memory", t_atom64_shared},
        {"double atomicAdd (exact sum)", t_atom_f64},
        {"IEEE binary64: 17 ops x 262k inputs, bit-exact vs CPU", t_f64_exact},
        {"__constant__ / __device__ globals + cuModuleGetGlobal", t_globals},
        {"managed, pinned, registered memory; memset; DtoD", t_memory_kinds},
        {"virtual memory management (reserve/create/map)", t_vmm},
        {"streams & events timing", t_events_streams},
        {"async copies: stream order, source reuse, honest capabilities", t_async_semantics},
        {"error paths & cuGetProcAddress", t_errors_and_queries},
        {"10k dependent launches", t_many_launches},
    };
    for (size_t i = 0; i < sizeof tests / sizeof tests[0]; i++) {
        if (only && !strstr(tests[i].name, only)) continue;
        int ok = tests[i].fn();
        printf("  [%s] %s\n", ok ? "PASS" : "FAIL", tests[i].name);
        if (ok) g_pass++;
        else g_fail++;
    }
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
