/*
 * test_softfloat.c — differential test of MetaCuda's IEEE binary64 soft-float.
 *
 * Oracle: the CPU's hardware FPU (Apple Silicon, IEEE 754) running each operation
 * under fesetround() in the matching rounding mode — an implementation that
 * shares no code with src/softfloat64.metal.
 * Device side: hand-written PTX, generated here, one kernel per (operation,
 * rounding modifier), so no compiler sits between the PTX instruction and the
 * test (add/sub/mul/div/fma/sqrt/rcp .rn/.rz/.rm/.rp, and the f64->f32,
 * s64/u64->f64, round-to-integral and f64->s64 conversions).
 * Inputs: an exhaustive grid of special values (zeros, subnormal and normal
 * boundaries, max finite, infinities, NaN, values one ulp around 1 and 2, ...)
 * followed by TestFloat-style random operands: boundary exponents, structured
 * significands (all ones, single bits, runs), nearby exponents for alignment,
 * near-cancellation pairs and fma addends close to -(a*b).
 *
 *   usage: test_softfloat [inputs-per-kernel]   (default 1048576)
 */
#pragma STDC FENV_ACCESS ON
#include <cuda.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { CUresult _r = (x); if (_r) { fprintf(stderr, "%s failed: %d\n", #x, _r); exit(2); } } while (0)

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t r64(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double bd(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t db(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

static const int MODES[4] = {FE_TONEAREST, FE_TOWARDZERO, FE_DOWNWARD, FE_UPWARD};
static const char *MNAME[4] = {"rn", "rz", "rm", "rp"};

enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_FMA, OP_SQRT, OP_RCP, OP_D2F, OP_S2D, OP_U2D, OP_RINT, OP_D2S64, OP_COUNT };
static const char *ONAME[OP_COUNT] = {"add", "sub", "mul", "div", "fma", "sqrt", "rcp", "cvt.f32.f64", "cvt.f64.s64", "cvt.f64.u64",
                                      "cvt.Xi.f64.f64", "cvt.Xi.s64.f64"};

/* ------------------------------------------------------------------ PTX */

static void emit_kernel(char *buf, size_t n, int op, int m) {
    char body[512];
    const char *M = MNAME[m];
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
        snprintf(body, sizeof body, "  %s.%s.f64 %%fd4, %%fd1, %%fd2;\n  st.global.f64 [%%rd9], %%fd4;\n", ONAME[op], M);
        break;
    case OP_FMA: snprintf(body, sizeof body, "  fma.%s.f64 %%fd4, %%fd1, %%fd2, %%fd3;\n  st.global.f64 [%%rd9], %%fd4;\n", M); break;
    case OP_SQRT: snprintf(body, sizeof body, "  sqrt.%s.f64 %%fd4, %%fd1;\n  st.global.f64 [%%rd9], %%fd4;\n", M); break;
    case OP_RCP: snprintf(body, sizeof body, "  rcp.%s.f64 %%fd4, %%fd1;\n  st.global.f64 [%%rd9], %%fd4;\n", M); break;
    case OP_D2F: snprintf(body, sizeof body, "  cvt.%s.f32.f64 %%f1, %%fd1;\n  st.global.f32 [%%rd9], %%f1;\n", M); break;
    case OP_S2D: snprintf(body, sizeof body, "  ld.global.u64 %%rd10, [%%rd3];\n  cvt.%s.f64.s64 %%fd4, %%rd10;\n  st.global.f64 [%%rd9], %%fd4;\n", M); break;
    case OP_U2D: snprintf(body, sizeof body, "  ld.global.u64 %%rd10, [%%rd3];\n  cvt.%s.f64.u64 %%fd4, %%rd10;\n  st.global.f64 [%%rd9], %%fd4;\n", M); break;
    case OP_RINT: snprintf(body, sizeof body, "  cvt.%si.f64.f64 %%fd4, %%fd1;\n  st.global.f64 [%%rd9], %%fd4;\n", M); break;
    default: snprintf(body, sizeof body, "  cvt.%si.s64.f64 %%rd10, %%fd1;\n  st.global.u64 [%%rd9], %%rd10;\n", M); break;
    }
    size_t l = strlen(buf);
    snprintf(buf + l, n - l,
             ".visible .entry k%d_%d(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 po, .param .u32 pn)\n{\n"
             "  .reg .pred %%p1;\n  .reg .b32 %%r<6>;\n  .reg .b64 %%rd<12>;\n  .reg .f64 %%fd<6>;\n  .reg .f32 %%f1;\n"
             "  mov.u32 %%r1, %%ctaid.x;\n  mov.u32 %%r2, %%ntid.x;\n  mov.u32 %%r3, %%tid.x;\n  mad.lo.s32 %%r4, %%r1, %%r2, %%r3;\n"
             "  ld.param.u32 %%r5, [pn];\n  setp.ge.u32 %%p1, %%r4, %%r5;\n  @%%p1 bra DONE;\n"
             "  mul.wide.u32 %%rd1, %%r4, 8;\n"
             "  ld.param.u64 %%rd2, [pa];\n  add.s64 %%rd3, %%rd2, %%rd1;\n  ld.global.f64 %%fd1, [%%rd3];\n"
             "  ld.param.u64 %%rd4, [pb];\n  add.s64 %%rd5, %%rd4, %%rd1;\n  ld.global.f64 %%fd2, [%%rd5];\n"
             "  ld.param.u64 %%rd6, [pc];\n  add.s64 %%rd7, %%rd6, %%rd1;\n  ld.global.f64 %%fd3, [%%rd7];\n"
             "  ld.param.u64 %%rd8, [po];\n  add.s64 %%rd9, %%rd8, %%rd1;\n"
             "%s"
             "DONE:\n  ret;\n}\n",
             op, m, body);
}

/* --------------------------------------------------------------- inputs */

static const double SPECIAL_POS[] = {0.0, 4.9406564584124654e-324, 2.2250738585072009e-308, 2.2250738585072014e-308,
                                     1.0, 1.0000000000000002, 0.99999999999999989, 2.0, 1.9999999999999998, 1.5, 3.0, 0.1,
                                     9007199254740992.0, 9007199254740994.0, 1.7976931348623157e308, INFINITY};
#define NSP_POS (int)(sizeof SPECIAL_POS / sizeof SPECIAL_POS[0])
static double special(int i) { /* 2*NSP_POS signed specials + quiet NaN */
    if (i == 2 * NSP_POS) return NAN;
    double v = SPECIAL_POS[i % NSP_POS];
    return i >= NSP_POS ? -v : v;
}
#define NSP (2 * NSP_POS + 1)

static uint64_t gen_sig(void) {
    switch (r64() % 8) {
    case 0: return 0;
    case 1: return 0xFFFFFFFFFFFFFull;
    case 2: return 1ull << (r64() % 52);
    case 3: return 0xFFFFFFFFFFFFFull >> (r64() % 52);                       /* low run of ones */
    case 4: return (0xFFFFFFFFFFFFFull << (r64() % 52)) & 0xFFFFFFFFFFFFFull; /* high run of ones */
    case 5: return (r64() & r64() & r64()) & 0xFFFFFFFFFFFFFull;             /* sparse */
    case 6: return 0x8000000000000ull | (r64() % 4);                          /* near half-way */
    default: return r64() & 0xFFFFFFFFFFFFFull;
    }
}
static int gen_exp(void) {
    static const int fixed[] = {0, 0, 1, 2, 1022, 1023, 1024, 1075, 1076, 970, 2045, 2046, 2046, 2047};
    if (r64() % 3) return 1 + (int)(r64() % 2046);
    return fixed[r64() % (sizeof fixed / sizeof fixed[0])];
}
static double gen_any(void) { return bd(((r64() & 1) << 63) | ((uint64_t)gen_exp() << 52) | gen_sig()); }
static double gen_near(double x) { /* an operand whose exponent is close to x's */
    int e = (int)((db(x) >> 52) & 0x7FF) + (int)(r64() % 9) - 4;
    if (e < 0) e = 0;
    if (e > 2046) e = 2046;
    return bd(((r64() & 1) << 63) | ((uint64_t)e << 52) | gen_sig());
}
static double ulp_step(double x, int k) { return bd(db(x) + (uint64_t)(int64_t)k); }

static void make_inputs(int op, int n, double *a, double *b, double *c) {
    int i = 0;
    /* exhaustive special grid */
    if (op == OP_FMA) {
        for (int x = 0; x < NSP && i < n; x++)
            for (int y = 0; y < NSP && i < n; y++)
                for (int z = 0; z < NSP && i < n; z++, i++) { a[i] = special(x); b[i] = special(y); c[i] = special(z); }
    } else {
        for (int x = 0; x < NSP && i < n; x++)
            for (int y = 0; y < NSP && i < n; y++, i++) { a[i] = special(x); b[i] = special(y); c[i] = 0; }
    }
    for (; i < n; i++) {
        a[i] = gen_any();
        int k = (int)(r64() % 4);
        b[i] = k == 0 ? gen_any() : k == 1 ? gen_near(a[i]) : k == 2 ? -ulp_step(a[i], (int)(r64() % 5) - 2) : ulp_step(a[i], (int)(r64() % 5) - 2);
        if (op == OP_FMA) {
            int kc = (int)(r64() % 4);
            if (kc == 0) c[i] = gen_any();
            else if (kc == 1) c[i] = gen_near(a[i] * b[i]);
            else { fesetround(FE_TONEAREST); c[i] = ulp_step(-(a[i] * b[i]), (int)(r64() % 7) - 3); } /* massive cancellation */
            if (r64() % 2) b[i] = gen_any(); /* keep a mix of unrelated a, b too */
        } else c[i] = 0;
        if (op == OP_S2D || op == OP_U2D) { uint64_t v = r64() >> (r64() % 64); if (r64() % 2) v = ~v; memcpy(&a[i], &v, 8); }
        if (op == OP_RINT || op == OP_D2S64) { /* stress the integer boundary: |x| around 0.5 .. 2^64 */
            int e = 1022 + (int)(r64() % 66);
            a[i] = r64() % 4 ? bd(((r64() & 1) << 63) | ((uint64_t)e << 52) | gen_sig()) : a[i];
        }
    }
}

/* ---------------------------------------------------------------- oracle */

static uint64_t oracle(int op, int m, double x, double y, double z) {
    volatile double a = x, b = y, c = z, r = 0;
    fesetround(MODES[m]);
    switch (op) {
    case OP_ADD: r = a + b; break;
    case OP_SUB: r = a - b; break;
    case OP_MUL: r = a * b; break;
    case OP_DIV: r = a / b; break;
    case OP_FMA: r = fma(a, b, c); break;
    case OP_SQRT: r = sqrt(a); break;
    case OP_RCP: r = 1.0 / a; break;
    case OP_D2F: { volatile float f = (float)a; fesetround(FE_TONEAREST); uint32_t u; memcpy(&u, (const void *)&f, 4); return u; }
    case OP_S2D: { int64_t v; memcpy(&v, &x, 8); volatile int64_t vv = v; r = (double)vv; break; }
    case OP_U2D: { uint64_t v; memcpy(&v, &x, 8); volatile uint64_t vv = v; r = (double)vv; break; }
    case OP_RINT: r = nearbyint(a); break;
    default: {
        double t = nearbyint(a);
        int64_t v;
        if (t != t) v = 0;
        else if (t >= 9223372036854775808.0) v = INT64_MAX;
        else if (t < -9223372036854775808.0) v = INT64_MIN;
        else v = (int64_t)t;
        fesetround(FE_TONEAREST);
        return (uint64_t)v;
    }
    }
    fesetround(FE_TONEAREST);
    double rr = r;
    return db(rr);
}

static int is_nan_result(int op, uint64_t v) {
    if (op == OP_D2F) return (v & 0x7FFFFFFFu) > 0x7F800000u;
    if (op == OP_D2S64) return 0;
    return (v & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1 << 20;
    /* sanity: the oracle really honours the rounding mode */
    if (oracle(OP_DIV, 1, 1.0, 3.0, 0) == oracle(OP_DIV, 3, 1.0, 3.0, 0)) { fprintf(stderr, "fesetround has no effect: oracle unusable\n"); return 2; }

    static char ptx[1 << 20];
    snprintf(ptx, sizeof ptx, ".version 7.0\n.target sm_80\n.address_size 64\n");
    for (int op = 0; op < OP_COUNT; op++)
        for (int m = 0; m < 4; m++) emit_kernel(ptx, sizeof ptx, op, m);
    CK(cuInit(0));
    CUmodule mod;
    CK(cuModuleLoadData(&mod, ptx));
    double *a = malloc((size_t)n * 8), *b = malloc((size_t)n * 8), *c = malloc((size_t)n * 8);
    uint64_t *out = malloc((size_t)n * 8);
    CUdeviceptr da, dbuf, dc, dout;
    CK(cuMemAlloc(&da, (size_t)n * 8)); CK(cuMemAlloc(&dbuf, (size_t)n * 8)); CK(cuMemAlloc(&dc, (size_t)n * 8)); CK(cuMemAlloc(&dout, (size_t)n * 8));
    long total_bad = 0, total = 0;
    printf("binary64 soft-float vs hardware FPU (fesetround), %d inputs per kernel\n", n);
    for (int op = 0; op < OP_COUNT; op++) {
        make_inputs(op, n, a, b, c);
        CK(cuMemcpyHtoD(da, a, (size_t)n * 8)); CK(cuMemcpyHtoD(dbuf, b, (size_t)n * 8)); CK(cuMemcpyHtoD(dc, c, (size_t)n * 8));
        printf("  %-15s", ONAME[op]);
        for (int m = 0; m < 4; m++) {
            char name[32];
            snprintf(name, sizeof name, "k%d_%d", op, m);
            CUfunction f;
            CK(cuModuleGetFunction(&f, mod, name));
            void *args[] = {&da, &dbuf, &dc, &dout, &n};
            CK(cuMemsetD8(dout, 0, (size_t)n * 8));
            CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL));
            CK(cuMemcpyDtoH(out, dout, (size_t)n * 8));
            long bad = 0;
            for (int i = 0; i < n; i++) {
                uint64_t want = oracle(op, m, a[i], b[i], c[i]);
                uint64_t got = op == OP_D2F ? (out[i] & 0xFFFFFFFFull) : out[i];
                if (got == want || (is_nan_result(op, want) && is_nan_result(op, got))) continue;
                if (bad++ < 3)
                    fprintf(stderr, "\n    MISMATCH %s.%s (%a, %a, %a): got 0x%016llx want 0x%016llx", ONAME[op], MNAME[m], a[i], b[i], c[i],
                            (unsigned long long)got, (unsigned long long)want);
            }
            printf("  %s:%s", MNAME[m], bad ? "FAIL" : "ok");
            total_bad += bad;
            total += n;
        }
        printf("\n");
    }
    printf("%ld / %ld results differ%s\n", total_bad, total, total_bad ? "" : " — bit-exact in all four rounding modes");
    return total_bad ? 1 : 0;
}
