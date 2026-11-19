/*
 * MetaCuda — PTX → Metal Shading Language translator.
 *
 * Parses a PTX module (as produced by nvcc or clang's NVPTX backend) and
 * emits one MSL compute kernel per `.entry`. The output is plain MSL source
 * that is compiled at runtime with -[MTLDevice newLibraryWithSource:].
 *
 * ABI contract between the translator and the runtime (see runtime.m):
 *   buffer(0)      : `constant uchar* __params` — the CUDA kernel parameter
 *                    block, laid out exactly like CUDA's (each param at its
 *                    natural/declared alignment), followed by a 24-byte hidden
 *                    block at `hidden_offset`: { u64 globals_base; u32 dyn_smem; u32 pad;
 *                    u64 lock_table } (lock_table: MC_LOCK_TABLE_BYTES used by 64-bit atomics).
 *   threadgroup(0) : `threadgroup uchar* __smem` (only if `uses_smem`) —
 *                    static shared variables at [0, smem_dyn_offset), dynamic
 *                    (`extern .shared`) memory starting at `smem_dyn_offset`.
 *   CUDA device pointers are Metal GPU virtual addresses (MTLBuffer.gpuAddress),
 *   so pointer arithmetic and pointers stored in memory work unchanged.
 */
#ifndef METACUDA_PTX_TO_METAL_H
#define METACUDA_PTX_TO_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MC_TRANSLATOR_VERSION "metacuda-ptx2msl-1.1"
#define MC_MAX_NAME 256
#define MC_HIDDEN_BYTES 24u
#define MC_LOCK_TABLE_BYTES (16384u * 4u)

typedef struct {
    char     name[MC_MAX_NAME];
    uint32_t offset;   /* byte offset inside the parameter block   */
    uint32_t size;     /* byte size of the parameter               */
    uint32_t align;    /* alignment used for layout                */
} mc_param_t;

typedef struct {
    char        name[MC_MAX_NAME];      /* PTX entry name (possibly mangled) */
    char        msl_name[MC_MAX_NAME];  /* MSL kernel function name          */
    mc_param_t *params;
    int         nparams;
    uint32_t    param_bytes;            /* size of the user parameter area   */
    uint32_t    hidden_offset;          /* offset of the hidden block         */
    uint32_t    total_param_bytes;      /* hidden_offset + MC_HIDDEN_BYTES    */
    int         uses_smem;              /* kernel takes threadgroup(0)        */
    uint32_t    smem_dyn_offset;        /* where extern shared memory starts  */
    uint32_t    static_smem;            /* bytes of static shared memory      */
    uint32_t    local_bytes;            /* per-thread local depot size        */
    uint32_t    reqntid[3];             /* .reqntid / .maxntid (0 = none)     */
    int         convergent;             /* uses barriers / warp intrinsics    */
} mc_kernel_t;

typedef struct {
    char     name[MC_MAX_NAME];
    uint64_t offset;   /* offset inside the module globals buffer */
    uint64_t size;
    int      is_const;
} mc_global_t;

typedef struct {
    char        *msl;            /* translated MSL source (NUL terminated) */
    size_t       msl_len;
    mc_kernel_t *kernels;
    int          nkernels;
    mc_global_t *globals;
    int          nglobals;
    uint8_t     *global_init;    /* initial image of all .global/.const data */
    uint64_t     globals_bytes;
    char        *warnings;       /* newline separated, may be NULL */
    char         error[1024];
} mc_module_t;

/* Returns 0 on success; on failure returns -1 and fills out->error. */
int  mc_translate_ptx(const char *ptx, size_t len, mc_module_t *out);
void mc_module_free(mc_module_t *m);

#ifdef __cplusplus
}
#endif
#endif
