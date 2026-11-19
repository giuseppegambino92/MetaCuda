/*
 * MetaCuda — CUDA Driver API implemented on Apple Metal.
 *
 *  - Device memory  : MTLBuffers created with newBufferWithBytesNoCopy over
 *                     page-aligned mmap'ed memory (MTLResourceStorageModeShared).
 *                     CUdeviceptr == MTLBuffer.gpuAddress (+offset), so device
 *                     pointers can be stored in memory and do arithmetic on the
 *                     GPU exactly like on CUDA. Host access goes through the
 *                     same physical pages (Unified Memory, zero copies).
 *  - Residency      : every allocation lives in an MTLResidencySet attached to
 *                     the command queue, so kernels may dereference any pointer.
 *  - Modules        : PTX (raw, fatbin, fatbin wrapper) → MSL (ptx_to_metal.c)
 *                     → MTLLibrary → MTLComputePipelineState, with an on-disk
 *                     cache (~/.cache/metalcuda) of the MSL and a Metal binary
 *                     archive of the compiled GPU code.
 *  - Execution      : launches are batched into one serial compute encoder
 *                     (with buffer memory barriers between dispatches) and
 *                     committed on synchronisation points.
 *  - Streams        : all streams share one in-order Metal queue. This is a
 *                     legal CUDA schedule (streams *may* run concurrently, they
 *                     are not required to).
 */
#define METACUDA_INTERNAL 1
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <CommonCrypto/CommonDigest.h>
#include <IOKit/IOKitLib.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../include/cuda.h"
#include "fatbin.h"
#include "mc_internal.h"
#include "ptx_to_metal.h"

#define MC_API __attribute__((visibility("default")))
#define PAGE 16384ull

/* ───────────────────────────── logging ───────────────────────────── */

int mc_log_level(void) {
    static int lvl = -1;
    if (lvl < 0) {
        const char *e = getenv("METALCUDA_LOG");
        lvl = e ? atoi(e) : 0;
        if (e && !*e) lvl = 1;
    }
    return lvl;
}
void mc_log(int level, const char *fmt, ...) {
    if (mc_log_level() < level) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[metalcuda] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}
#define LOGW(...) do { fprintf(stderr, "[metalcuda] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while (0)

/* ───────────────────────────── global state ───────────────────────────── */

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id g_rset; /* id<MTLResidencySet> (macOS 15+) */
static id<MTLSharedEvent> g_chain;
static uint64_t g_chain_val;
static id<MTLCommandBuffer> g_cb;
static id<MTLComputeCommandEncoder> g_enc;
static id<MTLCommandBuffer> g_last_cb;
static int g_ndisp;
static pthread_mutex_t g_mtx;
static int g_cores;
static char g_name[256];
static size_t g_total_mem;
static uint32_t g_max_tg_mem;
static _Atomic int g_sticky; /* sticky CUresult from asynchronous failures */
static char g_cache_dir[1024];
static int g_sync_each_launch;
static int g_initialized;
static uint64_t g_lock_table; /* GPU address of the 64-bit atomic lock table */

#define LOCK() pthread_mutex_lock(&g_mtx)
#define UNLOCK() pthread_mutex_unlock(&g_mtx)

typedef enum { ALLOC_DEVICE = 0, ALLOC_HOST, ALLOC_MANAGED, ALLOC_RESERVED, ALLOC_REGISTERED } alloc_kind_t;

typedef struct {
    uint64_t gpu;      /* gpuAddress of the MTLBuffer (+ offset for registered memory) */
    uint64_t size;     /* user visible size  */
    uint64_t maplen;   /* mapped length      */
    uint8_t *cpu;      /* CPU address of byte 0 */
    void *buf;         /* retained id<MTLBuffer> */
    int kind;
    int owns_memory;
} alloc_t;

static alloc_t **g_allocs; /* sorted by gpu */
static int g_nallocs, g_capallocs;
static int g_nhostlike;    /* number of host / managed / registered allocations */
static uint64_t g_bytes_allocated;

static int gpu_core_count(void) {
    int cores = 0;
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &it) != KERN_SUCCESS) return 0;
    io_object_t obj;
    while ((obj = IOIteratorNext(it))) {
        CFTypeRef v = IORegistryEntryCreateCFProperty(obj, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (v) {
            if (CFGetTypeID(v) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &cores);
            CFRelease(v);
        }
        IOObjectRelease(obj);
        if (cores) break;
    }
    IOObjectRelease(it);
    return cores;
}

static void init_cache_dir(void) {
    const char *e = getenv("METALCUDA_CACHE_DIR");
    if (e && *e) snprintf(g_cache_dir, sizeof g_cache_dir, "%s", e);
    else {
        const char *x = getenv("XDG_CACHE_HOME");
        const char *h = getenv("HOME");
        if (x && *x) snprintf(g_cache_dir, sizeof g_cache_dir, "%s/metalcuda", x);
        else snprintf(g_cache_dir, sizeof g_cache_dir, "%s/.cache/metalcuda", h ? h : "/tmp");
    }
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s", g_cache_dir);
    for (char *p = tmp + 1; *p; p++)
        if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    mkdir(tmp, 0755);
}

static void mc_do_init(void) {
    pthread_mutexattr_t a;
    pthread_mutexattr_init(&a);
    pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&g_mtx, &a);
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { LOGW("no Metal device available"); return; }
        g_queue = [g_dev newCommandQueueWithMaxCommandBufferCount:1024];
        g_queue.label = @"metalcuda";
        g_chain = [g_dev newSharedEvent];
        if (@available(macOS 15.0, *)) {
            MTLResidencySetDescriptor *d = [MTLResidencySetDescriptor new];
            d.initialCapacity = 256;
            d.label = @"metalcuda.allocations";
            NSError *err = nil;
            id<MTLResidencySet> rs = [g_dev newResidencySetWithDescriptor:d error:&err];
            if (rs) {
                [g_queue addResidencySet:rs];
                g_rset = rs;
            }
        }
        g_cores = gpu_core_count();
        if (!g_cores) g_cores = 8;
        snprintf(g_name, sizeof g_name, "%s (MetaCuda)", g_dev.name.UTF8String);
        g_total_mem = (size_t)g_dev.recommendedMaxWorkingSetSize;
        g_max_tg_mem = (uint32_t)g_dev.maxThreadgroupMemoryLength;
        const char *s = getenv("METALCUDA_SYNC");
        g_sync_each_launch = s && atoi(s) > 0;
        init_cache_dir();
        g_initialized = 1;
        mc_log(1, "initialised: %s, %d GPU cores, %.1f GiB working set, %u B threadgroup memory, residency sets: %s, cache: %s", g_name,
               g_cores, g_total_mem / 1073741824.0, g_max_tg_mem, g_rset ? "yes" : "no", g_cache_dir);
    }
}

static CUresult ensure_init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, mc_do_init);
    return g_initialized ? CUDA_SUCCESS : CUDA_ERROR_NO_DEVICE;
}
#define INIT() do { CUresult _r = ensure_init(); if (_r) return _r; } while (0)

/* ───────────────────────────── allocation table ───────────────────────────── */

static void table_insert(alloc_t *a) {
    if (g_nallocs == g_capallocs) {
        g_capallocs = g_capallocs ? g_capallocs * 2 : 64;
        g_allocs = (alloc_t **)realloc(g_allocs, (size_t)g_capallocs * sizeof(alloc_t *));
    }
    int lo = 0, hi = g_nallocs;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (g_allocs[mid]->gpu < a->gpu) lo = mid + 1;
        else hi = mid;
    }
    memmove(&g_allocs[lo + 1], &g_allocs[lo], (size_t)(g_nallocs - lo) * sizeof(alloc_t *));
    g_allocs[lo] = a;
    g_nallocs++;
    if (a->kind != ALLOC_DEVICE && a->kind != ALLOC_RESERVED) g_nhostlike++;
}
static void table_remove(alloc_t *a) {
    for (int i = 0; i < g_nallocs; i++)
        if (g_allocs[i] == a) {
            memmove(&g_allocs[i], &g_allocs[i + 1], (size_t)(g_nallocs - i - 1) * sizeof(alloc_t *));
            g_nallocs--;
            if (a->kind != ALLOC_DEVICE && a->kind != ALLOC_RESERVED) g_nhostlike--;
            return;
        }
}
static alloc_t *find_gpu(uint64_t p) {
    int lo = 0, hi = g_nallocs - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        alloc_t *a = g_allocs[mid];
        if (p < a->gpu) hi = mid - 1;
        else if (p >= a->gpu + (a->size ? a->size : 1)) lo = mid + 1;
        else return a;
    }
    return NULL;
}
static alloc_t *find_cpu(const void *ptr) {
    const uint8_t *p = (const uint8_t *)ptr;
    for (int i = 0; i < g_nallocs; i++) {
        alloc_t *a = g_allocs[i];
        if (a->kind == ALLOC_DEVICE || a->kind == ALLOC_RESERVED) continue;
        if (p >= a->cpu && p < a->cpu + (a->size ? a->size : 1)) return a;
    }
    return NULL;
}

/* Resolve any CUDA-visible address (device pointer, host-alloc pointer or plain
 * host memory) to a CPU pointer. *is_dev tells whether it was a known allocation. */
static void *resolve(uint64_t p, int *known) {
    alloc_t *a = find_gpu(p);
    if (a) { if (known) *known = 1; return a->cpu + (p - a->gpu); }
    a = g_nhostlike ? find_cpu((const void *)p) : NULL;
    if (known) *known = a != NULL;
    return (void *)p;
}

static void rset_add(id<MTLBuffer> b) {
    if (@available(macOS 15.0, *)) {
        if (g_rset) {
            id<MTLResidencySet> rs = g_rset;
            [rs addAllocation:b];
            [rs commit];
        }
    }
}
static void rset_remove(id<MTLBuffer> b) {
    if (@available(macOS 15.0, *)) {
        if (g_rset) {
            id<MTLResidencySet> rs = g_rset;
            [rs removeAllocation:b];
            [rs commit];
        }
    }
}

/* Wraps [base, base+len) (page aligned) in an MTLBuffer. */
static alloc_t *wrap_memory(void *base, size_t len, int kind, int owns) {
    @autoreleasepool {
        MTLResourceOptions opt = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        id<MTLBuffer> b;
        if (owns) {
            b = [g_dev newBufferWithBytesNoCopy:base length:len options:opt deallocator:^(void *p, NSUInteger l) { munmap(p, l); }];
        } else {
            b = [g_dev newBufferWithBytesNoCopy:base length:len options:opt deallocator:nil];
        }
        if (!b) return NULL;
        alloc_t *a = (alloc_t *)calloc(1, sizeof *a);
        a->buf = (void *)CFBridgingRetain(b);
        a->gpu = b.gpuAddress;
        a->cpu = (uint8_t *)base;
        a->size = len;
        a->maplen = len;
        a->kind = kind;
        a->owns_memory = owns;
        rset_add(b);
        return a;
    }
}

static CUresult mc_alloc(size_t size, int kind, alloc_t **out) {
    INIT();
    size_t len = (size_t)(((size ? size : 1) + PAGE - 1) / PAGE * PAGE);
    if (len > g_dev.maxBufferLength) return CUDA_ERROR_OUT_OF_MEMORY;
    void *mem = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (mem == MAP_FAILED) return CUDA_ERROR_OUT_OF_MEMORY;
    alloc_t *a = wrap_memory(mem, len, kind, 1);
    if (!a) { munmap(mem, len); return CUDA_ERROR_OUT_OF_MEMORY; }
    a->size = size;
    LOCK();
    table_insert(a);
    g_bytes_allocated += len;
    UNLOCK();
    mc_log(2, "alloc %zu bytes kind=%d gpu=0x%llx cpu=%p", size, kind, a->gpu, a->cpu);
    *out = a;
    return CUDA_SUCCESS;
}

static void mc_free_alloc(alloc_t *a) {
    LOCK();
    table_remove(a);
    g_bytes_allocated -= a->maplen;
    UNLOCK();
    id<MTLBuffer> b = CFBridgingRelease(a->buf);
    rset_remove(b);
    b = nil; /* deallocator unmaps owned memory */
    free(a);
}

/* ───────────────────────────── submission ───────────────────────────── */

static id<MTLComputeCommandEncoder> encoder(void) {
    if (!g_enc) {
        g_cb = [g_queue commandBuffer];
        g_cb.label = @"metalcuda.batch";
        if (g_chain_val) [g_cb encodeWaitForEvent:g_chain value:g_chain_val];
        g_enc = [g_cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        g_ndisp = 0;
    }
    return g_enc;
}

static void commit_locked(void) {
    if (!g_cb) return;
    if (g_enc) { [g_enc endEncoding]; g_enc = nil; }
    g_chain_val++;
    [g_cb encodeSignalEvent:g_chain value:g_chain_val];
    int nd = g_ndisp;
    [g_cb addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        mc_log(3, "command buffer: %d dispatches, GPU %.3f ms", nd, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
        if (cb.status == MTLCommandBufferStatusError) {
            LOGW("GPU command buffer failed: %s", cb.error.localizedDescription.UTF8String);
            int expected = 0;
            atomic_compare_exchange_strong(&g_sticky, &expected, (int)CUDA_ERROR_LAUNCH_FAILED);
        }
    }];
    [g_cb commit];
    g_last_cb = g_cb;
    g_cb = nil;
    g_ndisp = 0;
}

/* Waits for all submitted GPU work. */
static CUresult mc_sync(void) {
    if (!g_initialized) return CUDA_SUCCESS;
    id<MTLCommandBuffer> last;
    LOCK();
    commit_locked();
    last = g_last_cb;
    UNLOCK();
    if (last) {
        [last waitUntilCompleted];
        if (last.status == MTLCommandBufferStatusError) {
            int expected = 0;
            atomic_compare_exchange_strong(&g_sticky, &expected, (int)CUDA_ERROR_LAUNCH_FAILED);
        }
    }
    return (CUresult)atomic_load(&g_sticky);
}

/* ───────────────────────────── modules ───────────────────────────── */

struct CUfunc_st {
    struct CUmod_st *mod;
    mc_kernel_t *k;
    void *pso; /* id<MTLComputePipelineState> */
    int max_threads;
    int max_dyn_smem;
};

struct CUmod_st {
    mc_module_t ir;
    void *lib; /* id<MTLLibrary> */
    alloc_t *globals;
    struct CUfunc_st *funcs;
    int nfuncs;
    char hash[65];
};

static void sha256_hex(const void *data, size_t n, char out[65]) {
    unsigned char d[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_CTX c;
    CC_SHA256_Init(&c);
    CC_SHA256_Update(&c, MC_TRANSLATOR_VERSION, (CC_LONG)strlen(MC_TRANSLATOR_VERSION));
    CC_SHA256_Update(&c, data, (CC_LONG)n);
    CC_SHA256_Final(d, &c);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) snprintf(out + 2 * i, 3, "%02x", d[i]);
}

static int file_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

/* Floating-point contraction policy. PTX fixes the operation sequence (fma/mad
 * are explicit), so by default Metal is not allowed to fuse mul+add on its own:
 * results then match the PTX bit for bit. METALCUDA_FP_CONTRACT=fast lets the
 * Metal compiler contract (what ptxas may do for non-.rn ops). */
static const char *fp_contract_pragma(void) {
    const char *e = getenv("METALCUDA_FP_CONTRACT");
    if (e && !strcmp(e, "fast")) return "#pragma METAL fp contract(fast)\n";
    return "#pragma METAL fp contract(off)\n";
}

static CUresult build_module(const char *ptx, size_t len, struct CUmod_st **out) {
    INIT();
    LOCK();
    if (!g_lock_table) {
        alloc_t *lt;
        if (mc_alloc(MC_LOCK_TABLE_BYTES, ALLOC_DEVICE, &lt) == CUDA_SUCCESS) g_lock_table = lt->gpu; /* zero-filled by mmap */
    }
    UNLOCK();
    double t0 = CACurrentMediaTime();
    struct CUmod_st *m = (struct CUmod_st *)calloc(1, sizeof *m);
    const char *pragma = fp_contract_pragma();
    if (mc_translate_ptx(ptx, len, &m->ir)) {
        LOGW("PTX translation failed: %s", m->ir.error);
        free(m);
        return CUDA_ERROR_INVALID_PTX;
    }
    /* cache key = the exact Metal source that gets compiled (fp policy pragma + MSL),
     * so any translator or soft-float change invalidates stale entries */
    {
        size_t pl = strlen(pragma);
        char *keyed = (char *)malloc(m->ir.msl_len + pl);
        memcpy(keyed, pragma, pl);
        memcpy(keyed + pl, m->ir.msl, m->ir.msl_len);
        sha256_hex(keyed, m->ir.msl_len + pl, m->hash);
        free(keyed);
    }
    double t1 = CACurrentMediaTime();
    if (m->ir.warnings && mc_log_level() >= 1) fprintf(stderr, "[metalcuda] translator warnings:\n%s", m->ir.warnings);

    char msl_path[1200], arc_path[1200];
    snprintf(msl_path, sizeof msl_path, "%s/%s.metal", g_cache_dir, m->hash);
    snprintf(arc_path, sizeof arc_path, "%s/%s.mtlarchive", g_cache_dir, m->hash);
    if (!file_exists(msl_path)) {
        FILE *f = fopen(msl_path, "wb");
        if (f) { fputs(pragma, f); fwrite(m->ir.msl, 1, m->ir.msl_len, f); fclose(f); }
    }
    int warm = file_exists(arc_path);

    @autoreleasepool {
        MTLCompileOptions *opt = [MTLCompileOptions new];
        opt.languageVersion = MTLLanguageVersion3_2;
        if (@available(macOS 15.0, *)) opt.mathMode = MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opt.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        opt.preprocessorMacros = @{@"MC_NUM_SM" : [NSString stringWithFormat:@"%du", g_cores]};
        NSError *err = nil;
        NSString *src = [[NSString stringWithUTF8String:pragma]
            stringByAppendingString:[[NSString alloc] initWithBytes:m->ir.msl length:m->ir.msl_len encoding:NSUTF8StringEncoding]];
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) {
            LOGW("Metal compilation of translated PTX failed (MSL kept at %s):\n%s", msl_path, err.localizedDescription.UTF8String);
            mc_module_free(&m->ir);
            free(m);
            return CUDA_ERROR_INVALID_PTX;
        }
        double t2 = CACurrentMediaTime();
        m->lib = (void *)CFBridgingRetain(lib);

        /* binary archive: reuse compiled GPU binaries across processes */
        id<MTLBinaryArchive> archive = nil;
        MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
        if (warm) ad.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:arc_path]];
        archive = [g_dev newBinaryArchiveWithDescriptor:ad error:&err];
        if (!archive && warm) { /* corrupt / incompatible archive: start fresh */
            warm = 0;
            ad.url = nil;
            archive = [g_dev newBinaryArchiveWithDescriptor:ad error:&err];
        }

        m->nfuncs = m->ir.nkernels;
        m->funcs = (struct CUfunc_st *)calloc((size_t)(m->nfuncs ? m->nfuncs : 1), sizeof(struct CUfunc_st));
        int added = 0;
        for (int i = 0; i < m->nfuncs; i++) {
            mc_kernel_t *k = &m->ir.kernels[i];
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:k->msl_name]];
            if (!fn) { LOGW("kernel %s missing from compiled library", k->name); continue; }
            MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
            pd.computeFunction = fn;
            pd.label = [NSString stringWithUTF8String:k->name];
            uint32_t req = k->reqntid[0] ? k->reqntid[0] * (k->reqntid[1] ? k->reqntid[1] : 1) * (k->reqntid[2] ? k->reqntid[2] : 1) : 0;
            if (req && req <= 1024) pd.maxTotalThreadsPerThreadgroup = req;
            if (archive && warm) pd.binaryArchives = @[ archive ];
            id<MTLComputePipelineState> ps = [g_dev newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone reflection:nil error:&err];
            if (!ps) {
                LOGW("pipeline creation for %s failed: %s", k->name, err.localizedDescription.UTF8String);
                continue;
            }
            if (archive && !warm) {
                if ([archive addComputePipelineFunctionsWithDescriptor:pd error:&err]) added++;
            }
            struct CUfunc_st *f = &m->funcs[i];
            f->mod = m;
            f->k = k;
            f->pso = (void *)CFBridgingRetain(ps);
            f->max_threads = (int)ps.maxTotalThreadsPerThreadgroup;
            int avail = (int)g_max_tg_mem - (int)(k->uses_smem ? k->smem_dyn_offset : 0);
            f->max_dyn_smem = avail > 0 ? avail : 0;
            if (ps.threadExecutionWidth != 32) LOGW("warning: SIMD width is %lu, CUDA warp intrinsics assume 32", (unsigned long)ps.threadExecutionWidth);
        }
        if (archive && added) {
            NSError *se = nil;
            if (![archive serializeToURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:arc_path]] error:&se])
                mc_log(1, "could not write binary archive: %s", se.localizedDescription.UTF8String);
        }
        double t3 = CACurrentMediaTime();
        mc_log(1, "module %.12s: %d kernel(s), translate %.2f ms, MSL compile %.2f ms, pipelines %.2f ms (%s cache)", m->hash, m->nfuncs,
               (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, warm ? "warm" : "cold");
    }

    if (m->ir.globals_bytes) {
        CUresult r = mc_alloc((size_t)m->ir.globals_bytes, ALLOC_DEVICE, &m->globals);
        if (r) { mc_module_free(&m->ir); free(m); return r; }
        memcpy(m->globals->cpu, m->ir.global_init, (size_t)m->ir.globals_bytes);
    }
    *out = m;
    return CUDA_SUCCESS;
}

/* ───────────────────────────── error strings ───────────────────────────── */

static const struct { CUresult r; const char *name, *desc; } ERRS[] = {
    {CUDA_SUCCESS, "CUDA_SUCCESS", "no error"},
    {CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE", "invalid argument"},
    {CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY", "out of memory"},
    {CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED", "initialization error"},
    {CUDA_ERROR_DEINITIALIZED, "CUDA_ERROR_DEINITIALIZED", "driver shutting down"},
    {CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE", "no CUDA-capable device is detected"},
    {CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal"},
    {CUDA_ERROR_INVALID_IMAGE, "CUDA_ERROR_INVALID_IMAGE", "device kernel image is invalid"},
    {CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT", "invalid device context"},
    {CUDA_ERROR_NO_BINARY_FOR_GPU, "CUDA_ERROR_NO_BINARY_FOR_GPU", "no kernel image is available for execution on the device"},
    {CUDA_ERROR_INVALID_PTX, "CUDA_ERROR_INVALID_PTX", "a PTX JIT compilation failed"},
    {CUDA_ERROR_INVALID_SOURCE, "CUDA_ERROR_INVALID_SOURCE", "device kernel image is invalid"},
    {CUDA_ERROR_FILE_NOT_FOUND, "CUDA_ERROR_FILE_NOT_FOUND", "file not found"},
    {CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE", "invalid resource handle"},
    {CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND", "named symbol not found"},
    {CUDA_ERROR_NOT_READY, "CUDA_ERROR_NOT_READY", "device not ready"},
    {CUDA_ERROR_ILLEGAL_ADDRESS, "CUDA_ERROR_ILLEGAL_ADDRESS", "an illegal memory access was encountered"},
    {CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", "too many resources requested for launch"},
    {CUDA_ERROR_LAUNCH_TIMEOUT, "CUDA_ERROR_LAUNCH_TIMEOUT", "the launch timed out and was terminated"},
    {CUDA_ERROR_LAUNCH_FAILED, "CUDA_ERROR_LAUNCH_FAILED", "unspecified launch failure"},
    {CUDA_ERROR_NOT_PERMITTED, "CUDA_ERROR_NOT_PERMITTED", "operation not permitted"},
    {CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED", "operation not supported"},
    {CUDA_ERROR_UNKNOWN, "CUDA_ERROR_UNKNOWN", "unknown error"},
};

MC_API CUresult cuGetErrorString(CUresult e, const char **s) {
    for (size_t i = 0; i < sizeof ERRS / sizeof ERRS[0]; i++)
        if (ERRS[i].r == e) { *s = ERRS[i].desc; return CUDA_SUCCESS; }
    *s = NULL;
    return CUDA_ERROR_INVALID_VALUE;
}
MC_API CUresult cuGetErrorName(CUresult e, const char **s) {
    for (size_t i = 0; i < sizeof ERRS / sizeof ERRS[0]; i++)
        if (ERRS[i].r == e) { *s = ERRS[i].name; return CUDA_SUCCESS; }
    *s = NULL;
    return CUDA_ERROR_INVALID_VALUE;
}

/* ───────────────────────────── init / device ───────────────────────────── */

MC_API CUresult cuInit(unsigned int flags) {
    (void)flags;
    return ensure_init();
}
MC_API CUresult cuDriverGetVersion(int *v) {
    if (!v) return CUDA_ERROR_INVALID_VALUE;
    *v = CUDA_VERSION;
    return CUDA_SUCCESS;
}
MC_API CUresult cuDeviceGetCount(int *count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    *count = ensure_init() == CUDA_SUCCESS ? 1 : 0;
    return CUDA_SUCCESS;
}
MC_API CUresult cuDeviceGet(CUdevice *d, int ordinal) {
    INIT();
    if (!d) return CUDA_ERROR_INVALID_VALUE;
    if (ordinal != 0) return CUDA_ERROR_INVALID_DEVICE;
    *d = 0;
    return CUDA_SUCCESS;
}
MC_API CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    INIT();
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    snprintf(name, (size_t)len, "%s", g_name);
    return CUDA_SUCCESS;
}
MC_API CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    INIT();
    if (!uuid || dev) return CUDA_ERROR_INVALID_VALUE;
    uint64_t id = g_dev.registryID;
    memset(uuid->bytes, 0, 16);
    memcpy(uuid->bytes, "MetaCuda", 8);
    memcpy(uuid->bytes + 8, &id, 8);
    return CUDA_SUCCESS;
}
MC_API CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) { return cuDeviceGetUuid(uuid, dev); }
MC_API CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    INIT();
    if (!bytes || dev) return CUDA_ERROR_INVALID_VALUE;
    *bytes = g_total_mem;
    return CUDA_SUCCESS;
}
MC_API CUresult cuDeviceTotalMem(unsigned int *bytes, CUdevice dev) {
    size_t b;
    CUresult r = cuDeviceTotalMem_v2(&b, dev);
    if (!r) *bytes = b > 0xFFFFFFFFu ? 0xFFFFFFFFu : (unsigned)b;
    return r;
}

int mc_shared_mem_per_block(void) { return (int)g_max_tg_mem - 32; }

MC_API CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    INIT();
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    if (dev) return CUDA_ERROR_INVALID_DEVICE;
    int v = 0;
    switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: v = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: v = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: v = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: v = 64; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: v = 2147483647; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: v = 65535; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: v = 65535; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: v = mc_shared_mem_per_block(); break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: v = (int)g_max_tg_mem; break;
    case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: v = 65536; break;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE: v = 32; break;
    case CU_DEVICE_ATTRIBUTE_MAX_PITCH: v = 2147483647; break;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: v = 65536; break;
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: v = 1398000; break; /* kHz */
    case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: v = 256; break;
    case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: v = 0; break;          /* no copy/compute overlap */
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: v = g_cores; break;
    case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_INTEGRATED: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: v = 0; break;   /* one in-order queue */
    case CU_DEVICE_ATTRIBUTE_ECC_ENABLED: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_TCC_DRIVER: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: v = 3200000; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: v = 384; break;
    case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: v = 8 * 1024 * 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: v = 2048; break;
    case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: v = 8; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD: case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO: v = 64; break;
    case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: case CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: v = 32; break;
    case CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK: v = 32; break;
    case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: v = 0; break;
    default: v = 0; break;
    }
    *pi = v;
    return CUDA_SUCCESS;
}
MC_API CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    if (!major || !minor) return CUDA_ERROR_INVALID_VALUE;
    cuDeviceGetAttribute(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    return cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
}

/* ───────────────────────────── contexts ───────────────────────────── */

struct CUctx_st { int device; unsigned flags; int refcount; };
static struct CUctx_st g_primary = {0, 0, 0};
static __thread CUcontext t_ctx_stack[16];
static __thread int t_ctx_depth;

static void set_current(CUcontext c) {
    if (t_ctx_depth == 0) t_ctx_depth = 1;
    t_ctx_stack[t_ctx_depth - 1] = c;
}

MC_API CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    INIT();
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (dev) return CUDA_ERROR_INVALID_DEVICE;
    g_primary.refcount++;
    *pctx = &g_primary;
    return CUDA_SUCCESS;
}
MC_API CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) { if (dev) return CUDA_ERROR_INVALID_DEVICE; if (g_primary.refcount) g_primary.refcount--; return CUDA_SUCCESS; }
MC_API CUresult cuDevicePrimaryCtxRelease(CUdevice dev) { return cuDevicePrimaryCtxRelease_v2(dev); }
MC_API CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) { if (dev) return CUDA_ERROR_INVALID_DEVICE; mc_sync(); return CUDA_SUCCESS; }
MC_API CUresult cuDevicePrimaryCtxReset(CUdevice dev) { return cuDevicePrimaryCtxReset_v2(dev); }
MC_API CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    if (dev) return CUDA_ERROR_INVALID_DEVICE;
    if (flags) *flags = g_primary.flags;
    if (active) *active = g_primary.refcount > 0;
    return CUDA_SUCCESS;
}
MC_API CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) { if (dev) return CUDA_ERROR_INVALID_DEVICE; g_primary.flags = flags; return CUDA_SUCCESS; }
MC_API CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) { return cuDevicePrimaryCtxSetFlags_v2(dev, flags); }

MC_API CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    INIT();
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (dev) return CUDA_ERROR_INVALID_DEVICE;
    struct CUctx_st *c = (struct CUctx_st *)calloc(1, sizeof *c);
    c->flags = flags;
    c->refcount = 1;
    *pctx = c;
    if (t_ctx_depth < 16) t_ctx_stack[t_ctx_depth++] = c;
    return CUDA_SUCCESS;
}
MC_API CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) { return cuCtxCreate_v2(pctx, flags, dev); }
MC_API CUresult cuCtxDestroy_v2(CUcontext ctx) {
    if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
    mc_sync();
    for (int i = 0; i < t_ctx_depth; i++) if (t_ctx_stack[i] == ctx) t_ctx_stack[i] = NULL;
    while (t_ctx_depth && !t_ctx_stack[t_ctx_depth - 1]) t_ctx_depth--;
    if (ctx != &g_primary) free(ctx);
    return CUDA_SUCCESS;
}
MC_API CUresult cuCtxDestroy(CUcontext ctx) { return cuCtxDestroy_v2(ctx); }
MC_API CUresult cuCtxSetCurrent(CUcontext ctx) { INIT(); set_current(ctx); return CUDA_SUCCESS; }
MC_API CUresult cuCtxGetCurrent(CUcontext *pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = t_ctx_depth ? t_ctx_stack[t_ctx_depth - 1] : NULL;
    return CUDA_SUCCESS;
}
MC_API CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    INIT();
    if (t_ctx_depth >= 16) return CUDA_ERROR_INVALID_VALUE;
    t_ctx_stack[t_ctx_depth++] = ctx;
    return CUDA_SUCCESS;
}
MC_API CUresult cuCtxPushCurrent(CUcontext ctx) { return cuCtxPushCurrent_v2(ctx); }
MC_API CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
    if (!t_ctx_depth) return CUDA_ERROR_INVALID_CONTEXT;
    CUcontext c = t_ctx_stack[--t_ctx_depth];
    if (pctx) *pctx = c;
    return CUDA_SUCCESS;
}
MC_API CUresult cuCtxPopCurrent(CUcontext *pctx) { return cuCtxPopCurrent_v2(pctx); }
MC_API CUresult cuCtxGetDevice(CUdevice *d) { INIT(); if (!d) return CUDA_ERROR_INVALID_VALUE; *d = 0; return CUDA_SUCCESS; }
MC_API CUresult cuCtxSynchronize(void) { INIT(); return mc_sync(); }
MC_API CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) { (void)ctx; if (version) *version = 3020; return CUDA_SUCCESS; }
MC_API CUresult cuCtxGetFlags(unsigned int *flags) { if (flags) *flags = g_primary.flags; return CUDA_SUCCESS; }
static size_t g_limits[3] = {1024, 1048576, 8388608};
MC_API CUresult cuCtxSetLimit(CUlimit limit, size_t value) { if ((unsigned)limit < 3) g_limits[limit] = value; return CUDA_SUCCESS; }
MC_API CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    if (!pvalue || (unsigned)limit >= 3) return CUDA_ERROR_UNSUPPORTED_LIMIT;
    *pvalue = g_limits[limit];
    return CUDA_SUCCESS;
}
MC_API CUresult cuCtxSetCacheConfig(CUfunc_cache c) { (void)c; return CUDA_SUCCESS; }
MC_API CUresult cuCtxGetStreamPriorityRange(int *lo, int *hi) { if (lo) *lo = 0; if (hi) *hi = 0; return CUDA_SUCCESS; }

/* ───────────────────────────── streams & events ───────────────────────────── */

struct CUstream_st { unsigned flags; int priority; };
struct CUevent_st { void *cb; double host_time; unsigned flags; int recorded; };

MC_API CUresult cuStreamCreate(CUstream *ph, unsigned int flags) {
    INIT();
    if (!ph) return CUDA_ERROR_INVALID_VALUE;
    struct CUstream_st *s = (struct CUstream_st *)calloc(1, sizeof *s);
    s->flags = flags;
    *ph = s;
    return CUDA_SUCCESS;
}
MC_API CUresult cuStreamCreateWithPriority(CUstream *ph, unsigned int flags, int priority) {
    CUresult r = cuStreamCreate(ph, flags);
    if (!r) (*ph)->priority = priority;
    return r;
}
static int is_special_stream(CUstream s) { return s == NULL || s == CU_STREAM_LEGACY || s == CU_STREAM_PER_THREAD; }
MC_API CUresult cuStreamDestroy_v2(CUstream s) {
    if (is_special_stream(s)) return CUDA_ERROR_INVALID_HANDLE;
    free(s);
    return CUDA_SUCCESS;
}
MC_API CUresult cuStreamDestroy(CUstream s) { return cuStreamDestroy_v2(s); }
MC_API CUresult cuStreamSynchronize(CUstream s) { (void)s; INIT(); return mc_sync(); }
MC_API CUresult cuStreamSynchronize_ptsz(CUstream s) { return cuStreamSynchronize(s); }
MC_API CUresult cuStreamQuery(CUstream s) {
    (void)s;
    INIT();
    LOCK();
    int pending = g_cb != nil;
    id<MTLCommandBuffer> last = g_last_cb;
    UNLOCK();
    if (pending) return CUDA_ERROR_NOT_READY;
    if (last && last.status < MTLCommandBufferStatusCompleted) return CUDA_ERROR_NOT_READY;
    return CUDA_SUCCESS;
}
MC_API CUresult cuStreamWaitEvent(CUstream s, CUevent e, unsigned int flags) {
    (void)s; (void)e; (void)flags; /* single in-order queue: already ordered */
    return CUDA_SUCCESS;
}
MC_API CUresult cuStreamGetFlags(CUstream s, unsigned int *flags) { if (flags) *flags = is_special_stream(s) ? 0 : s->flags; return CUDA_SUCCESS; }
MC_API CUresult cuStreamGetPriority(CUstream s, int *p) { if (p) *p = is_special_stream(s) ? 0 : s->priority; return CUDA_SUCCESS; }

MC_API CUresult cuEventCreate(CUevent *ph, unsigned int flags) {
    INIT();
    if (!ph) return CUDA_ERROR_INVALID_VALUE;
    struct CUevent_st *e = (struct CUevent_st *)calloc(1, sizeof *e);
    e->flags = flags;
    *ph = e;
    return CUDA_SUCCESS;
}
MC_API CUresult cuEventRecord(CUevent e, CUstream s) {
    (void)s;
    INIT();
    if (!e) return CUDA_ERROR_INVALID_HANDLE;
    LOCK();
    commit_locked();
    id<MTLCommandBuffer> cb = g_last_cb;
    if (e->cb) CFRelease(e->cb);
    e->cb = cb ? (void *)CFBridgingRetain(cb) : NULL;
    e->host_time = CACurrentMediaTime();
    e->recorded = 1;
    UNLOCK();
    return CUDA_SUCCESS;
}
MC_API CUresult cuEventRecord_ptsz(CUevent e, CUstream s) { return cuEventRecord(e, s); }
MC_API CUresult cuEventRecordWithFlags(CUevent e, CUstream s, unsigned int f) { (void)f; return cuEventRecord(e, s); }
MC_API CUresult cuEventSynchronize(CUevent e) {
    if (!e) return CUDA_ERROR_INVALID_HANDLE;
    if (e->cb) [(__bridge id<MTLCommandBuffer>)e->cb waitUntilCompleted];
    return (CUresult)atomic_load(&g_sticky);
}
MC_API CUresult cuEventQuery(CUevent e) {
    if (!e) return CUDA_ERROR_INVALID_HANDLE;
    if (e->cb && ((__bridge id<MTLCommandBuffer>)e->cb).status < MTLCommandBufferStatusCompleted) return CUDA_ERROR_NOT_READY;
    return CUDA_SUCCESS;
}
static double event_time(struct CUevent_st *e) {
    double t = e->host_time;
    if (e->cb) {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)e->cb;
        [cb waitUntilCompleted];
        if (cb.GPUEndTime > t) t = cb.GPUEndTime;
    }
    return t;
}
MC_API CUresult cuEventElapsedTime(float *ms, CUevent a, CUevent b) {
    if (!ms || !a || !b) return CUDA_ERROR_INVALID_VALUE;
    if (!a->recorded || !b->recorded) return CUDA_ERROR_INVALID_HANDLE;
    if ((a->flags | b->flags) & CU_EVENT_DISABLE_TIMING) return CUDA_ERROR_INVALID_HANDLE;
    *ms = (float)((event_time(b) - event_time(a)) * 1000.0);
    return CUDA_SUCCESS;
}
MC_API CUresult cuEventDestroy_v2(CUevent e) {
    if (!e) return CUDA_ERROR_INVALID_HANDLE;
    if (e->cb) CFRelease(e->cb);
    free(e);
    return CUDA_SUCCESS;
}
MC_API CUresult cuEventDestroy(CUevent e) { return cuEventDestroy_v2(e); }

/* ───────────────────────────── memory ───────────────────────────── */

MC_API CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    INIT();
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    if (bytesize == 0) { *dptr = 0; return CUDA_ERROR_INVALID_VALUE; }
    alloc_t *a;
    CUresult r = mc_alloc(bytesize, ALLOC_DEVICE, &a);
    if (r) return r;
    *dptr = a->gpu;
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemAlloc(CUdeviceptr *dptr, size_t n) { return cuMemAlloc_v2(dptr, n); }
MC_API CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pitch, size_t w, size_t h, unsigned int esz) {
    (void)esz;
    if (!pitch) return CUDA_ERROR_INVALID_VALUE;
    *pitch = (w + 255) / 256 * 256;
    return cuMemAlloc_v2(dptr, *pitch * h);
}
MC_API CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pitch, size_t w, size_t h, unsigned int esz) { return cuMemAllocPitch_v2(dptr, pitch, w, h, esz); }

/* Managed (and host) memory: the *CPU* address is returned so that host code
 * can dereference it directly. Kernel arguments pointing into such ranges are
 * rewritten to the GPU address at launch (see translate_params). */
MC_API CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    (void)flags;
    INIT();
    if (!dptr || !bytesize) return CUDA_ERROR_INVALID_VALUE;
    alloc_t *a;
    CUresult r = mc_alloc(bytesize, ALLOC_MANAGED, &a);
    if (r) return r;
    *dptr = (CUdeviceptr)(uintptr_t)a->cpu;
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int flags) {
    (void)flags;
    INIT();
    if (!pp) return CUDA_ERROR_INVALID_VALUE;
    alloc_t *a;
    CUresult r = mc_alloc(bytesize, ALLOC_HOST, &a);
    if (r) return r;
    *pp = a->cpu;
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemAllocHost_v2(void **pp, size_t n) { return cuMemHostAlloc(pp, n, 0); }
MC_API CUresult cuMemAllocHost(void **pp, size_t n) { return cuMemHostAlloc(pp, n, 0); }

MC_API CUresult cuMemFree_v2(CUdeviceptr dptr) {
    INIT();
    if (!dptr) return CUDA_SUCCESS;
    mc_sync(); /* cuMemFree is synchronising: in-flight kernels may still use it */
    LOCK();
    alloc_t *a = find_gpu(dptr);
    if (!a || a->gpu != dptr) a = g_nhostlike ? find_cpu((void *)dptr) : NULL;
    UNLOCK();
    if (!a) return CUDA_ERROR_INVALID_VALUE;
    mc_free_alloc(a);
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemFree(CUdeviceptr dptr) { return cuMemFree_v2(dptr); }
MC_API CUresult cuMemFreeHost(void *p) { return cuMemFree_v2((CUdeviceptr)(uintptr_t)p); }

MC_API CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pd, void *p, unsigned int flags) {
    (void)flags;
    INIT();
    if (!pd) return CUDA_ERROR_INVALID_VALUE;
    LOCK();
    alloc_t *a = find_cpu(p);
    UNLOCK();
    if (!a) return CUDA_ERROR_INVALID_VALUE;
    *pd = a->gpu + (uint64_t)((uint8_t *)p - a->cpu);
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemHostGetDevicePointer(CUdeviceptr *pd, void *p, unsigned int flags) { return cuMemHostGetDevicePointer_v2(pd, p, flags); }

MC_API CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int flags) {
    (void)flags;
    INIT();
    if (!p || !bytesize) return CUDA_ERROR_INVALID_VALUE;
    LOCK();
    alloc_t *existing = find_cpu(p);
    UNLOCK();
    if (existing) return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    uintptr_t base = (uintptr_t)p & ~(uintptr_t)(PAGE - 1);
    uintptr_t end = ((uintptr_t)p + bytesize + PAGE - 1) & ~(uintptr_t)(PAGE - 1);
    alloc_t *a = wrap_memory((void *)base, end - base, ALLOC_REGISTERED, 0);
    if (!a) return CUDA_ERROR_NOT_SUPPORTED;
    uint64_t delta = (uint64_t)((uintptr_t)p - base);
    a->gpu += delta;
    a->cpu += delta;
    a->size = bytesize;
    LOCK();
    table_insert(a);
    UNLOCK();
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemHostRegister(void *p, size_t n, unsigned int f) { return cuMemHostRegister_v2(p, n, f); }
MC_API CUresult cuMemHostUnregister(void *p) {
    mc_sync();
    LOCK();
    alloc_t *a = find_cpu(p);
    UNLOCK();
    if (!a || a->kind != ALLOC_REGISTERED) return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    mc_free_alloc(a);
    return CUDA_SUCCESS;
}

MC_API CUresult cuMemGetInfo_v2(size_t *free_b, size_t *total) {
    INIT();
    size_t used = (size_t)g_dev.currentAllocatedSize;
    if (total) *total = g_total_mem;
    if (free_b) *free_b = used < g_total_mem ? g_total_mem - used : 0;
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemGetInfo(size_t *f, size_t *t) { return cuMemGetInfo_v2(f, t); }

MC_API CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    INIT();
    LOCK();
    alloc_t *a = find_gpu(dptr);
    uint64_t base = 0;
    if (a) base = a->gpu;
    else if ((a = find_cpu((void *)dptr))) base = (uint64_t)(uintptr_t)a->cpu;
    UNLOCK();
    if (!a) return CUDA_ERROR_NOT_FOUND;
    if (pbase) *pbase = base;
    if (psize) *psize = (size_t)a->size;
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemGetAddressRange(CUdeviceptr *b, size_t *s, CUdeviceptr d) { return cuMemGetAddressRange_v2(b, s, d); }

/* --- copies: wait for the GPU, then plain CPU copies over unified memory --- */

static CUresult copy_generic(uint64_t dst, uint64_t src, size_t n) {
    INIT();
    if (!n) return CUDA_SUCCESS;
    CUresult r = mc_sync();
    if (r) return r;
    LOCK();
    void *d = resolve(dst, NULL);
    void *s = resolve(src, NULL);
    UNLOCK();
    memmove(d, s, n);
    return CUDA_SUCCESS;
}
static CUresult check_dev(uint64_t p) {
    LOCK();
    int known = 0;
    resolve(p, &known);
    UNLOCK();
    return known ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

MC_API CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t n) { return copy_generic(dst, src, n); }
MC_API CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream s) { (void)s; return copy_generic(dst, src, n); }
MC_API CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n) {
    INIT();
    if (n && check_dev(dst)) return CUDA_ERROR_INVALID_VALUE;
    return copy_generic(dst, (uint64_t)(uintptr_t)src, n);
}
MC_API CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n) {
    INIT();
    if (n && check_dev(src)) return CUDA_ERROR_INVALID_VALUE;
    return copy_generic((uint64_t)(uintptr_t)dst, src, n);
}
MC_API CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t n) {
    INIT();
    if (n && (check_dev(dst) || check_dev(src))) return CUDA_ERROR_INVALID_VALUE;
    return copy_generic(dst, src, n);
}
MC_API CUresult cuMemcpyHtoD(CUdeviceptr d, const void *s, size_t n) { return cuMemcpyHtoD_v2(d, s, n); }
MC_API CUresult cuMemcpyDtoH(void *d, CUdeviceptr s, size_t n) { return cuMemcpyDtoH_v2(d, s, n); }
MC_API CUresult cuMemcpyDtoD(CUdeviceptr d, CUdeviceptr s, size_t n) { return cuMemcpyDtoD_v2(d, s, n); }
MC_API CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr d, const void *s, size_t n, CUstream st) { (void)st; return cuMemcpyHtoD_v2(d, s, n); }
MC_API CUresult cuMemcpyDtoHAsync_v2(void *d, CUdeviceptr s, size_t n, CUstream st) { (void)st; return cuMemcpyDtoH_v2(d, s, n); }
MC_API CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream st) { (void)st; return cuMemcpyDtoD_v2(d, s, n); }
MC_API CUresult cuMemcpyHtoDAsync(CUdeviceptr d, const void *s, size_t n, CUstream st) { return cuMemcpyHtoDAsync_v2(d, s, n, st); }
MC_API CUresult cuMemcpyDtoHAsync(void *d, CUdeviceptr s, size_t n, CUstream st) { return cuMemcpyDtoHAsync_v2(d, s, n, st); }
MC_API CUresult cuMemcpyDtoDAsync(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream st) { return cuMemcpyDtoDAsync_v2(d, s, n, st); }

static CUresult memset_generic(uint64_t dst, const void *pattern, size_t esz, size_t count) {
    INIT();
    if (!count) return CUDA_SUCCESS;
    if (check_dev(dst)) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = mc_sync();
    if (r) return r;
    LOCK();
    uint8_t *d = (uint8_t *)resolve(dst, NULL);
    UNLOCK();
    if (esz == 1) memset(d, *(const uint8_t *)pattern, count);
    else if (esz == 2) { uint16_t v; memcpy(&v, pattern, 2); for (size_t i = 0; i < count; i++) memcpy(d + 2 * i, &v, 2); }
    else { uint32_t v; memcpy(&v, pattern, 4); if (!v) memset(d, 0, count * 4); else for (size_t i = 0; i < count; i++) memcpy(d + 4 * i, &v, 4); }
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemsetD8_v2(CUdeviceptr d, unsigned char v, size_t n) { return memset_generic(d, &v, 1, n); }
MC_API CUresult cuMemsetD16_v2(CUdeviceptr d, unsigned short v, size_t n) { return memset_generic(d, &v, 2, n); }
MC_API CUresult cuMemsetD32_v2(CUdeviceptr d, unsigned int v, size_t n) { return memset_generic(d, &v, 4, n); }
MC_API CUresult cuMemsetD8(CUdeviceptr d, unsigned char v, size_t n) { return memset_generic(d, &v, 1, n); }
MC_API CUresult cuMemsetD16(CUdeviceptr d, unsigned short v, size_t n) { return memset_generic(d, &v, 2, n); }
MC_API CUresult cuMemsetD32(CUdeviceptr d, unsigned int v, size_t n) { return memset_generic(d, &v, 4, n); }
MC_API CUresult cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream s) { (void)s; return memset_generic(d, &v, 1, n); }
MC_API CUresult cuMemsetD32Async(CUdeviceptr d, unsigned int v, size_t n, CUstream s) { (void)s; return memset_generic(d, &v, 4, n); }
MC_API CUresult cuMemPrefetchAsync(CUdeviceptr d, size_t n, CUdevice dev, CUstream s) { (void)d; (void)n; (void)dev; (void)s; return CUDA_SUCCESS; }
MC_API CUresult cuMemAdvise(CUdeviceptr d, size_t n, int advice, CUdevice dev) { (void)d; (void)n; (void)advice; (void)dev; return CUDA_SUCCESS; }

/* --- virtual memory management: a reserved range is one lazily-committed
 *     buffer, so cuMemMap only has to validate the range. --- */

struct mc_vmm_handle { size_t size; };

MC_API CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
    (void)alignment; (void)addr; (void)flags;
    INIT();
    if (!ptr || !size) return CUDA_ERROR_INVALID_VALUE;
    alloc_t *a;
    CUresult r = mc_alloc(size, ALLOC_RESERVED, &a);
    if (r) return r;
    *ptr = a->gpu;
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) { (void)size; return cuMemFree_v2(ptr); }
MC_API CUresult cuMemCreate(CUmemGenericAllocationHandle *h, size_t size, const CUmemAllocationProp *prop, unsigned long long flags) {
    (void)prop; (void)flags;
    if (!h || !size) return CUDA_ERROR_INVALID_VALUE;
    struct mc_vmm_handle *x = (struct mc_vmm_handle *)calloc(1, sizeof *x);
    x->size = size;
    *h = (CUmemGenericAllocationHandle)(uintptr_t)x;
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemRelease(CUmemGenericAllocationHandle h) { free((void *)(uintptr_t)h); return CUDA_SUCCESS; }
MC_API CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle h, unsigned long long flags) {
    (void)offset; (void)flags;
    INIT();
    if (!h) return CUDA_ERROR_INVALID_HANDLE;
    LOCK();
    alloc_t *a = find_gpu(ptr);
    UNLOCK();
    if (!a || a->kind != ALLOC_RESERVED || ptr + size > a->gpu + a->maplen) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    INIT();
    mc_sync();
    LOCK();
    alloc_t *a = find_gpu(ptr);
    UNLOCK();
    if (!a || a->kind != ALLOC_RESERVED) return CUDA_ERROR_INVALID_VALUE;
    uint8_t *c = a->cpu + (ptr - a->gpu);
    uintptr_t b = ((uintptr_t)c + PAGE - 1) & ~(uintptr_t)(PAGE - 1);
    uintptr_t e = ((uintptr_t)c + size) & ~(uintptr_t)(PAGE - 1);
    if (e > b) madvise((void *)b, e - b, MADV_FREE);
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count) {
    (void)ptr; (void)size; (void)desc; (void)count;
    return CUDA_SUCCESS;
}
MC_API CUresult cuMemGetAllocationGranularity(size_t *g, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags o) {
    (void)prop; (void)o;
    if (!g) return CUDA_ERROR_INVALID_VALUE;
    *g = 2 * 1024 * 1024;
    return CUDA_SUCCESS;
}

MC_API CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attr, CUdeviceptr ptr) {
    INIT();
    if (!data) return CUDA_ERROR_INVALID_VALUE;
    LOCK();
    alloc_t *a = find_gpu(ptr);
    int by_cpu = 0;
    if (!a && (a = find_cpu((void *)ptr))) by_cpu = 1;
    UNLOCK();
    if (!a) return CUDA_ERROR_INVALID_VALUE;
    uint64_t off = by_cpu ? ptr - (uint64_t)(uintptr_t)a->cpu : ptr - a->gpu;
    switch (attr) {
    case CU_POINTER_ATTRIBUTE_CONTEXT: *(CUcontext *)data = &g_primary; break;
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
        *(unsigned *)data = (a->kind == ALLOC_HOST || a->kind == ALLOC_REGISTERED) ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
        break;
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER: *(CUdeviceptr *)data = a->gpu + off; break;
    case CU_POINTER_ATTRIBUTE_HOST_POINTER: *(void **)data = a->cpu + off; break;
    case CU_POINTER_ATTRIBUTE_IS_MANAGED: *(unsigned *)data = a->kind == ALLOC_MANAGED; break;
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: *(int *)data = 0; break;
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR: *(CUdeviceptr *)data = by_cpu ? (uint64_t)(uintptr_t)a->cpu : a->gpu; break;
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE: *(size_t *)data = (size_t)a->size; break;
    case CU_POINTER_ATTRIBUTE_MAPPED: *(int *)data = 1; break;
    default: return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

int mc_pointer_info(const void *p, int *is_device, int *is_managed, void **host, uint64_t *dev) {
    if (ensure_init()) return 0;
    LOCK();
    alloc_t *a = find_gpu((uint64_t)(uintptr_t)p);
    int by_cpu = 0;
    if (!a && (a = find_cpu(p))) by_cpu = 1;
    UNLOCK();
    if (!a) return 0;
    uint64_t off = by_cpu ? (uint64_t)((const uint8_t *)p - a->cpu) : (uint64_t)(uintptr_t)p - a->gpu;
    if (is_device) *is_device = a->kind == ALLOC_DEVICE || a->kind == ALLOC_RESERVED;
    if (is_managed) *is_managed = a->kind == ALLOC_MANAGED;
    if (host) *host = a->cpu + off;
    if (dev) *dev = a->gpu + off;
    return 1;
}

/* ───────────────────────────── modules API ───────────────────────────── */

static CUresult load_image(CUmodule *mod, const void *image, size_t size) {
    INIT();
    if (!mod || !image) return CUDA_ERROR_INVALID_VALUE;
    char err[512] = "";
    int arch = 0;
    char *ptx = mc_extract_ptx(image, size, err, sizeof err, &arch);
    if (!ptx) {
        LOGW("cannot load module: %s", err);
        return CUDA_ERROR_NO_BINARY_FOR_GPU;
    }
    if (arch) mc_log(2, "using embedded PTX for compute_%d", arch);
    struct CUmod_st *m;
    CUresult r = build_module(ptx, strlen(ptx), &m);
    free(ptx);
    if (r) return r;
    *mod = m;
    return CUDA_SUCCESS;
}

MC_API CUresult cuModuleLoadData(CUmodule *mod, const void *image) { return load_image(mod, image, 0); }
MC_API CUresult cuModuleLoadDataEx(CUmodule *mod, const void *image, unsigned int n, CUjit_option *opts, void **vals) {
    CUresult r = load_image(mod, image, 0);
    /* honour info/error log buffers so JIT-error reporting code works */
    for (unsigned i = 0; opts && vals && i < n; i++) {
        if ((opts[i] == CU_JIT_INFO_LOG_BUFFER || opts[i] == CU_JIT_ERROR_LOG_BUFFER) && vals[i]) ((char *)vals[i])[0] = 0;
    }
    return r;
}
MC_API CUresult cuModuleLoadFatBinary(CUmodule *mod, const void *fatCubin) { return load_image(mod, fatCubin, 0); }
MC_API CUresult cuModuleLoad(CUmodule *mod, const char *fname) {
    if (!fname) return CUDA_ERROR_INVALID_VALUE;
    FILE *f = fopen(fname, "rb");
    if (!f) return CUDA_ERROR_FILE_NOT_FOUND;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = 0;
    fclose(f);
    CUresult r = load_image(mod, buf, got);
    free(buf);
    return r;
}
MC_API CUresult cuModuleUnload(CUmodule m) {
    if (!m) return CUDA_ERROR_INVALID_HANDLE;
    mc_sync();
    for (int i = 0; i < m->nfuncs; i++) if (m->funcs[i].pso) CFRelease(m->funcs[i].pso);
    if (m->lib) CFRelease(m->lib);
    if (m->globals) mc_free_alloc(m->globals);
    mc_module_free(&m->ir);
    free(m->funcs);
    free(m);
    return CUDA_SUCCESS;
}
MC_API CUresult cuModuleGetFunction(CUfunction *hf, CUmodule m, const char *name) {
    if (!hf || !m || !name) return CUDA_ERROR_INVALID_VALUE;
    for (int i = 0; i < m->nfuncs; i++)
        if (m->funcs[i].k && !strcmp(m->funcs[i].k->name, name)) {
            if (!m->funcs[i].pso) return CUDA_ERROR_INVALID_PTX;
            *hf = &m->funcs[i];
            return CUDA_SUCCESS;
        }
    return CUDA_ERROR_NOT_FOUND;
}
MC_API CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule m, const char *name) {
    if (!m || !name) return CUDA_ERROR_INVALID_VALUE;
    for (int i = 0; i < m->ir.nglobals; i++)
        if (!strcmp(m->ir.globals[i].name, name)) {
            if (dptr) *dptr = m->globals->gpu + m->ir.globals[i].offset;
            if (bytes) *bytes = (size_t)m->ir.globals[i].size;
            return CUDA_SUCCESS;
        }
    return CUDA_ERROR_NOT_FOUND;
}
MC_API CUresult cuModuleGetGlobal(CUdeviceptr *d, unsigned *b, CUmodule m, const char *n) {
    size_t s;
    CUresult r = cuModuleGetGlobal_v2(d, &s, m, n);
    if (!r && b) *b = (unsigned)s;
    return r;
}

MC_API CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attr, CUfunction f) {
    if (!pi || !f) return CUDA_ERROR_INVALID_VALUE;
    switch (attr) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = f->max_threads; break;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: *pi = (int)f->k->static_smem; break;
    case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: *pi = 0; break;
    case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: *pi = (int)f->k->local_bytes; break;
    case CU_FUNC_ATTRIBUTE_NUM_REGS: *pi = 32; break;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION: case CU_FUNC_ATTRIBUTE_BINARY_VERSION: *pi = 80; break;
    case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: *pi = 0; break;
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: *pi = f->max_dyn_smem; break;
    case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: *pi = -1; break;
    default: return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}
MC_API CUresult cuFuncSetAttribute(CUfunction f, CUfunction_attribute attr, int value) {
    if (!f) return CUDA_ERROR_INVALID_HANDLE;
    if (attr == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
        int avail = (int)g_max_tg_mem - (int)(f->k->uses_smem ? f->k->smem_dyn_offset : 0);
        if (value > avail) {
            LOGW("kernel %s requests %d B dynamic shared memory; Apple GPU threadgroup memory limit allows %d B", f->k->name, value, avail);
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    return CUDA_SUCCESS;
}
MC_API CUresult cuFuncSetCacheConfig(CUfunction f, CUfunc_cache c) { (void)f; (void)c; return CUDA_SUCCESS; }
MC_API CUresult cuFuncSetSharedMemConfig(CUfunction f, int c) { (void)f; (void)c; return CUDA_SUCCESS; }

/* ───────────────────────────── launch ───────────────────────────── */

/* Host / managed pointers passed as kernel arguments are rewritten to GPU addresses. */
static void translate_params(uint8_t *block, uint32_t n) {
    if (!g_nhostlike) return;
    for (uint32_t off = 0; off + 8 <= n; off += 8) {
        uint64_t v;
        memcpy(&v, block + off, 8);
        if (!v) continue;
        alloc_t *a = find_cpu((const void *)v);
        if (a) {
            uint64_t g = a->gpu + (v - (uint64_t)(uintptr_t)a->cpu);
            memcpy(block + off, &g, 8);
        }
    }
}

static CUresult launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shmem,
                       void **params, void **extra) {
    INIT();
    if (!f || !f->pso) return CUDA_ERROR_INVALID_HANDLE;
    if (!gx || !gy || !gz || !bx || !by || !bz) return CUDA_ERROR_INVALID_VALUE;
    uint64_t threads = (uint64_t)bx * by * bz;
    if (threads > 1024 || bz > 64) return CUDA_ERROR_INVALID_VALUE;
    if (threads > (uint64_t)f->max_threads) {
        LOGW("launch of %s with %llu threads/block exceeds the Metal pipeline limit of %d", f->k->name, (unsigned long long)threads, f->max_threads);
        return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    }
    mc_kernel_t *k = f->k;
    uint32_t total = k->total_param_bytes;
    uint8_t stackbuf[4096];
    uint8_t *block = total <= sizeof stackbuf ? stackbuf : (uint8_t *)malloc(total);
    memset(block, 0, total);
    if (params) {
        for (int i = 0; i < k->nparams; i++) {
            if (!params[i]) { if (block != stackbuf) free(block); return CUDA_ERROR_INVALID_VALUE; }
            memcpy(block + k->params[i].offset, params[i], k->params[i].size);
        }
    } else if (extra) {
        void *buf = NULL;
        size_t sz = 0;
        for (int i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
            if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = extra[i + 1];
            else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) sz = *(size_t *)extra[i + 1];
        }
        if (!buf) { if (block != stackbuf) free(block); return CUDA_ERROR_INVALID_VALUE; }
        memcpy(block, buf, sz < k->param_bytes ? sz : k->param_bytes);
    } else if (k->nparams) {
        if (block != stackbuf) free(block);
        return CUDA_ERROR_INVALID_VALUE;
    }
    uint32_t tg = 0;
    if (k->uses_smem) {
        tg = (k->smem_dyn_offset + shmem + 15u) & ~15u;
        if (tg > g_max_tg_mem) {
            LOGW("launch of %s needs %u B of shared memory (static %u + dynamic %u); Apple GPUs provide %u B per threadgroup", k->name, tg,
                 k->static_smem, shmem, g_max_tg_mem);
            if (block != stackbuf) free(block);
            return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        }
    }
    LOCK();
    translate_params(block, k->param_bytes);
    uint64_t gbase = f->mod->globals ? f->mod->globals->gpu : 0;
    memcpy(block + k->hidden_offset, &gbase, 8);
    memcpy(block + k->hidden_offset + 8, &shmem, 4);
    memcpy(block + k->hidden_offset + 16, &g_lock_table, 8);
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = encoder();
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)f->pso];
        if (total <= 4096) [enc setBytes:block length:total atIndex:0];
        else {
            id<MTLBuffer> pb = [g_dev newBufferWithBytes:block length:total options:MTLResourceStorageModeShared];
            [enc setBuffer:pb offset:0 atIndex:0];
        }
        if (k->uses_smem) [enc setThreadgroupMemoryLength:tg < 16 ? 16 : tg atIndex:0];
        if (!g_rset) { /* pre-macOS 15 fallback: make every allocation resident for this dispatch */
            for (int i = 0; i < g_nallocs; i++)
                [enc useResource:(__bridge id<MTLBuffer>)g_allocs[i]->buf usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(bx, by, bz)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        g_ndisp++;
        if (g_ndisp >= 256) commit_locked();
    }
    UNLOCK();
    if (block != stackbuf) free(block);
    mc_log(2, "launch %s grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%u", k->name, gx, gy, gz, bx, by, bz, shmem);
    if (g_sync_each_launch) return mc_sync();
    return (CUresult)atomic_load(&g_sticky);
}

MC_API CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz,
                               unsigned int shmem, CUstream s, void **params, void **extra) {
    (void)s;
    return launch(f, gx, gy, gz, bx, by, bz, shmem, params, extra);
}
MC_API CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by,
                                    unsigned int bz, unsigned int shmem, CUstream s, void **params, void **extra) {
    return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shmem, s, params, extra);
}
MC_API CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by,
                                          unsigned int bz, unsigned int shmem, CUstream s, void **params) {
    return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shmem, s, params, NULL);
}

MC_API CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *n, CUfunction f, int bs, size_t dsm) {
    if (!n || !f || bs <= 0) return CUDA_ERROR_INVALID_VALUE;
    int by_threads = 2048 / bs;
    int by_smem = 32;
    uint32_t need = f->k->uses_smem ? f->k->smem_dyn_offset + (uint32_t)dsm : 0;
    if (need) by_smem = (int)(g_max_tg_mem / need);
    *n = by_threads < by_smem ? by_threads : by_smem;
    if (*n < 1) *n = 0;
    return CUDA_SUCCESS;
}
MC_API CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *n, CUfunction f, int bs, size_t dsm, unsigned flags) {
    (void)flags;
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(n, f, bs, dsm);
}
MC_API CUresult cuOccupancyMaxPotentialBlockSize(int *minGrid, int *blockSize, CUfunction f, void *fn, size_t dsm, int limit) {
    (void)fn; (void)dsm;
    if (!f) return CUDA_ERROR_INVALID_HANDLE;
    int bs = f->max_threads;
    if (limit > 0 && limit < bs) bs = limit;
    if (blockSize) *blockSize = bs;
    if (minGrid) *minGrid = g_cores * (2048 / bs);
    return CUDA_SUCCESS;
}

/* ───────────────────────────── entry-point lookup ───────────────────────────── */

static void *self_handle(void) {
    static void *h;
    if (!h) {
        Dl_info i;
        if (dladdr((void *)&cuInit, &i)) h = dlopen(i.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
        if (!h) h = RTLD_DEFAULT;
    }
    return h;
}

MC_API CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, uint64_t flags, CUdriverProcAddressQueryResult *status) {
    (void)cudaVersion;
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
    char name[256];
    void *f = NULL;
    const char *suffixes_ptds[] = {"_v3_ptsz", "_v2_ptsz", "_ptsz", "_v3_ptds", "_v2_ptds", "_ptds"};
    if (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
        for (size_t i = 0; i < sizeof suffixes_ptds / sizeof suffixes_ptds[0] && !f; i++) {
            snprintf(name, sizeof name, "%s%s", symbol, suffixes_ptds[i]);
            f = dlsym(self_handle(), name);
        }
    const char *suffixes[] = {"_v3", "_v2", ""};
    for (size_t i = 0; i < 3 && !f; i++) {
        snprintf(name, sizeof name, "%s%s", symbol, suffixes[i]);
        f = dlsym(self_handle(), name);
    }
    /* only hand out symbols that live in this library */
    if (f) {
        Dl_info a, b;
        if (!dladdr(f, &a) || !dladdr((void *)&cuInit, &b) || a.dli_fbase != b.dli_fbase) f = NULL;
    }
    *pfn = f;
    if (status) *status = f ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    if (!f) mc_log(2, "cuGetProcAddress: %s not available", symbol);
    return f ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}
MC_API CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, uint64_t flags, CUdriverProcAddressQueryResult *status) {
    return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, status);
}
MC_API CUresult cuGetExportTable(const void **table, const CUuuid *id) {
    (void)id;
    if (table) *table = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* internal helpers for the runtime layer */
CUresult mc_sync_all(void) { return mc_sync(); }
const char *mc_device_name(void) { ensure_init(); return g_name; }
int mc_gpu_cores(void) { ensure_init(); return g_cores; }
size_t mc_total_mem(void) { ensure_init(); return g_total_mem; }
CUresult mc_function_info(CUfunction f, int *max_threads, int *static_smem, int *local_bytes) {
    if (!f) return CUDA_ERROR_INVALID_HANDLE;
    if (max_threads) *max_threads = f->max_threads;
    if (static_smem) *static_smem = (int)f->k->static_smem;
    if (local_bytes) *local_bytes = (int)f->k->local_bytes;
    return CUDA_SUCCESS;
}
