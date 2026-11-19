/*
 * MetaCuda — device-side CUDA language support for clang (`-x cuda -nocudainc`).
 *
 * Provides the qualifiers, built-in variables and the common device intrinsics
 * (atomics, warp shuffles/votes, fast math) on top of clang's NVVM builtins, so
 * that regular CUDA C++ compiles to PTX without NVIDIA's headers. The PTX is
 * then executed on Apple GPUs by libmetalcuda.
 */
#ifndef METACUDA_DEVICE_H
#define METACUDA_DEVICE_H

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __restrict__ __restrict

#include "vector_types.h"
#include "__clang_cuda_builtin_vars.h"

#define MC_DEV static __device__ __forceinline__

/* ---- synchronisation ---- */
MC_DEV void __syncwarp(unsigned mask = 0xffffffffu) { __nvvm_bar_warp_sync(mask); }
MC_DEV int __syncthreads_count(int p) { return __nvvm_bar0_popc(p); }
MC_DEV int __syncthreads_and(int p) { return __nvvm_bar0_and(p); }
MC_DEV int __syncthreads_or(int p) { return __nvvm_bar0_or(p); }
MC_DEV void __threadfence(void) { __nvvm_membar_gl(); }
MC_DEV void __threadfence_block(void) { __nvvm_membar_cta(); }
MC_DEV void __threadfence_system(void) { __nvvm_membar_sys(); }

/* ---- atomics ---- */
MC_DEV float atomicAdd(float *p, float v) { return __nvvm_atom_add_gen_f(p, v); }
MC_DEV int atomicAdd(int *p, int v) { return __nvvm_atom_add_gen_i(p, v); }
MC_DEV unsigned atomicAdd(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_add_gen_i((int *)p, (int)v); }
MC_DEV unsigned long long atomicAdd(unsigned long long *p, unsigned long long v) { return (unsigned long long)__nvvm_atom_add_gen_ll((long long *)p, (long long)v); }
MC_DEV int atomicSub(int *p, int v) { return __nvvm_atom_add_gen_i(p, -v); }
MC_DEV unsigned atomicSub(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_add_gen_i((int *)p, -(int)v); }
MC_DEV int atomicMax(int *p, int v) { return __nvvm_atom_max_gen_i(p, v); }
MC_DEV unsigned atomicMax(unsigned *p, unsigned v) { return __nvvm_atom_max_gen_ui(p, v); }
MC_DEV int atomicMin(int *p, int v) { return __nvvm_atom_min_gen_i(p, v); }
MC_DEV unsigned atomicMin(unsigned *p, unsigned v) { return __nvvm_atom_min_gen_ui(p, v); }
MC_DEV int atomicExch(int *p, int v) { return __nvvm_atom_xchg_gen_i(p, v); }
MC_DEV unsigned atomicExch(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_xchg_gen_i((int *)p, (int)v); }
MC_DEV float atomicExch(float *p, float v) { return __builtin_bit_cast(float, __nvvm_atom_xchg_gen_i((int *)p, __builtin_bit_cast(int, v))); }
MC_DEV int atomicCAS(int *p, int c, int v) { return __nvvm_atom_cas_gen_i(p, c, v); }
MC_DEV unsigned atomicCAS(unsigned *p, unsigned c, unsigned v) { return (unsigned)__nvvm_atom_cas_gen_i((int *)p, (int)c, (int)v); }
MC_DEV unsigned atomicInc(unsigned *p, unsigned v) { return __nvvm_atom_inc_gen_ui(p, v); }
MC_DEV unsigned atomicDec(unsigned *p, unsigned v) { return __nvvm_atom_dec_gen_ui(p, v); }
MC_DEV int atomicAnd(int *p, int v) { return __nvvm_atom_and_gen_i(p, v); }
MC_DEV int atomicOr(int *p, int v) { return __nvvm_atom_or_gen_i(p, v); }
MC_DEV int atomicXor(int *p, int v) { return __nvvm_atom_xor_gen_i(p, v); }
MC_DEV unsigned atomicOr(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_or_gen_i((int *)p, (int)v); }

/* ---- warp intrinsics ---- */
#define MC_SHFL_C(width) ((32 - (width)) << 8)
MC_DEV int __shfl_sync(unsigned m, int v, int src, int w = 32) { return __nvvm_shfl_sync_idx_i32(m, v, src, MC_SHFL_C(w) | 0x1f); }
MC_DEV int __shfl_down_sync(unsigned m, int v, unsigned d, int w = 32) { return __nvvm_shfl_sync_down_i32(m, v, (int)d, MC_SHFL_C(w) | 0x1f); }
MC_DEV int __shfl_up_sync(unsigned m, int v, unsigned d, int w = 32) { return __nvvm_shfl_sync_up_i32(m, v, (int)d, MC_SHFL_C(w)); }
MC_DEV int __shfl_xor_sync(unsigned m, int v, int l, int w = 32) { return __nvvm_shfl_sync_bfly_i32(m, v, l, MC_SHFL_C(w) | 0x1f); }
MC_DEV unsigned __shfl_sync(unsigned m, unsigned v, int src, int w = 32) { return (unsigned)__shfl_sync(m, (int)v, src, w); }
MC_DEV unsigned __shfl_down_sync(unsigned m, unsigned v, unsigned d, int w = 32) { return (unsigned)__shfl_down_sync(m, (int)v, d, w); }
MC_DEV unsigned __shfl_xor_sync(unsigned m, unsigned v, int l, int w = 32) { return (unsigned)__shfl_xor_sync(m, (int)v, l, w); }
MC_DEV float __shfl_sync(unsigned m, float v, int src, int w = 32) { return __nvvm_shfl_sync_idx_f32(m, v, src, MC_SHFL_C(w) | 0x1f); }
MC_DEV float __shfl_down_sync(unsigned m, float v, unsigned d, int w = 32) { return __nvvm_shfl_sync_down_f32(m, v, (int)d, MC_SHFL_C(w) | 0x1f); }
MC_DEV float __shfl_up_sync(unsigned m, float v, unsigned d, int w = 32) { return __nvvm_shfl_sync_up_f32(m, v, (int)d, MC_SHFL_C(w)); }
MC_DEV float __shfl_xor_sync(unsigned m, float v, int l, int w = 32) { return __nvvm_shfl_sync_bfly_f32(m, v, l, MC_SHFL_C(w) | 0x1f); }
MC_DEV unsigned __ballot_sync(unsigned m, int p) { return __nvvm_vote_ballot_sync(m, p); }
MC_DEV int __any_sync(unsigned m, int p) { return __nvvm_vote_any_sync(m, p); }
MC_DEV int __all_sync(unsigned m, int p) { return __nvvm_vote_all_sync(m, p); }
MC_DEV unsigned __activemask(void) { unsigned r; asm volatile("activemask.b32 %0;" : "=r"(r)); return r; }
MC_DEV unsigned __reduce_add_sync(unsigned m, unsigned v) { return __nvvm_redux_sync_add(v, m); }

/* ---- integer intrinsics ---- */
MC_DEV int __popc(unsigned x) { int r; asm("popc.b32 %0, %1;" : "=r"(r) : "r"(x)); return r; }
MC_DEV int __popcll(unsigned long long x) { int r; asm("popc.b64 %0, %1;" : "=r"(r) : "l"(x)); return r; }
MC_DEV int __clz(int x) { int r; asm("clz.b32 %0, %1;" : "=r"(r) : "r"(x)); return r; }
MC_DEV int __clzll(long long x) { int r; asm("clz.b64 %0, %1;" : "=r"(r) : "l"(x)); return r; }
MC_DEV unsigned __brev(unsigned x) { unsigned r; asm("brev.b32 %0, %1;" : "=r"(r) : "r"(x)); return r; }
MC_DEV int __ffs(int x) { return x == 0 ? 0 : __popc((unsigned)((x & -x) - 1)) + 1; }
MC_DEV unsigned __byte_perm(unsigned a, unsigned b, unsigned s) { return __nvvm_prmt(a, b, s); }
MC_DEV unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned s) { unsigned r; asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(s)); return r; }
MC_DEV unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned s) { unsigned r; asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(s)); return r; }
MC_DEV int __mul24(int a, int b) { return __nvvm_mul24_i(a, b); }
MC_DEV int __mulhi(int a, int b) { return __nvvm_mulhi_i(a, b); }
MC_DEV unsigned __umulhi(unsigned a, unsigned b) { return __nvvm_mulhi_ui(a, b); }
MC_DEV int __dp4a(int a, int b, int c) { int r; asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
MC_DEV unsigned __sad(int a, int b, unsigned c) { return __nvvm_sad_i(a, b, c); }

/* ---- float math (no libdevice needed) ---- */
MC_DEV float __int_as_float_mc(int x) { return __builtin_bit_cast(float, x); }
MC_DEV float __int_as_float(int x) { return __builtin_bit_cast(float, x); }
MC_DEV int __float_as_int(float x) { return __builtin_bit_cast(int, x); }
MC_DEV unsigned __float_as_uint(float x) { return __builtin_bit_cast(unsigned, x); }
MC_DEV float __uint_as_float(unsigned x) { return __builtin_bit_cast(float, x); }
MC_DEV float sqrtf(float x) { return __builtin_sqrtf(x); }
MC_DEV float rsqrtf(float x) { return __nvvm_rsqrt_approx_f(x); }
MC_DEV float fabsf(float x) { return __builtin_fabsf(x); }
MC_DEV float fminf(float a, float b) { return __builtin_fminf(a, b); }
MC_DEV float fmaxf(float a, float b) { return __builtin_fmaxf(a, b); }
MC_DEV float fmaf(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
MC_DEV float floorf(float x) { return __builtin_floorf(x); }
MC_DEV float ceilf(float x) { return __builtin_ceilf(x); }
MC_DEV float truncf(float x) { return __builtin_truncf(x); }
MC_DEV float rintf(float x) { return __builtin_rintf(x); }
MC_DEV float __expf(float x) { return __nvvm_ex2_approx_f(x * 1.4426950408889634f); }
MC_DEV float expf(float x) { return __expf(x); }
MC_DEV float exp2f(float x) { return __nvvm_ex2_approx_f(x); }
MC_DEV float __logf(float x) { return __nvvm_lg2_approx_f(x) * 0.6931471805599453f; }
MC_DEV float logf(float x) { return __logf(x); }
MC_DEV float log2f(float x) { return __nvvm_lg2_approx_f(x); }
MC_DEV float __sinf(float x) { return __nvvm_sin_approx_f(x); }
MC_DEV float __cosf(float x) { return __nvvm_cos_approx_f(x); }
MC_DEV float sinf(float x) { return __sinf(x); }
MC_DEV float cosf(float x) { return __cosf(x); }
MC_DEV float tanhf(float x) { float e = __expf(2.0f * x); return (e - 1.0f) / (e + 1.0f); }
MC_DEV float __fdividef(float a, float b) { return __nvvm_div_approx_f(a, b); }
MC_DEV float __saturatef(float x) { return __nvvm_saturate_f(x); }
MC_DEV float __fmaf_rn(float a, float b, float c) { return __nvvm_fma_rn_f(a, b, c); }
MC_DEV int __float2int_rn(float x) { return __nvvm_f2i_rn(x); }
MC_DEV int __float2int_rz(float x) { return __nvvm_f2i_rz(x); }
MC_DEV double sqrt(double x) { return __builtin_sqrt(x); }
MC_DEV double fabs(double x) { return __builtin_fabs(x); }
MC_DEV double fma(double a, double b, double c) { return __builtin_fma(a, b, c); }

template <typename T> MC_DEV T __ldg(const T *p) { return *p; }
template <typename T> MC_DEV T min(T a, T b) { return a < b ? a : b; }
template <typename T> MC_DEV T max(T a, T b) { return a > b ? a : b; }

extern "C" __device__ int printf(const char *, ...);

#endif /* METACUDA_DEVICE_H */
