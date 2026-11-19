/*
 * MetaCuda — CUDA Runtime API on top of the Driver API.
 *
 * Besides the documented cuda* functions this implements the private ABI that
 * nvcc/clang-generated host code calls:
 *   __cudaRegisterFatBinary / __cudaRegisterFatBinaryEnd / __cudaUnregisterFatBinary
 *   __cudaRegisterFunction / __cudaRegisterVar / __cudaRegisterManagedVar
 *   __cudaPushCallConfiguration / __cudaPopCallConfiguration   (CUDA >= 9.2 `<<<>>>`)
 *   cudaConfigureCall / cudaSetupArgument / cudaLaunch          (legacy `<<<>>>`)
 * so that a program compiled from .cu sources runs without modification.
 */
#define METACUDA_INTERNAL 1
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/cuda.h"
#include "../include/cuda_runtime_api.h"
#include "mc_internal.h"

#define MC_API __attribute__((visibility("default")))

static __thread cudaError_t t_last_error = cudaSuccess;
static pthread_mutex_t g_rt_mtx = PTHREAD_MUTEX_INITIALIZER;

static cudaError_t map_err(CUresult r) {
    switch (r) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_NO_DEVICE: return cudaErrorNoDevice;
    case CUDA_ERROR_INVALID_DEVICE: return cudaErrorInvalidDevice;
    case CUDA_ERROR_INVALID_IMAGE: return cudaErrorInvalidKernelImage;
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return cudaErrorNoKernelImageForDevice;
    case CUDA_ERROR_INVALID_PTX: return cudaErrorInvalidPtx;
    case CUDA_ERROR_INVALID_HANDLE: return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_NOT_FOUND: return cudaErrorSymbolNotFound;
    case CUDA_ERROR_NOT_READY: return cudaErrorNotReady;
    case CUDA_ERROR_ILLEGAL_ADDRESS: return cudaErrorIllegalAddress;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return cudaErrorLaunchOutOfResources;
    case CUDA_ERROR_LAUNCH_TIMEOUT: return cudaErrorLaunchTimeout;
    case CUDA_ERROR_LAUNCH_FAILED: return cudaErrorLaunchFailure;
    case CUDA_ERROR_NOT_SUPPORTED: return cudaErrorNotSupported;
    case CUDA_ERROR_NOT_PERMITTED: return cudaErrorNotPermitted;
    default: return cudaErrorUnknown;
    }
}
static cudaError_t ret(cudaError_t e) {
    if (e != cudaSuccess) t_last_error = e;
    return e;
}
#define RET(r) return ret(map_err(r))

/* ───────────────────────────── errors ───────────────────────────── */

static const struct { cudaError_t e; const char *name, *str; } RT_ERRS[] = {
    {cudaSuccess, "cudaSuccess", "no error"},
    {cudaErrorInvalidValue, "cudaErrorInvalidValue", "invalid argument"},
    {cudaErrorMemoryAllocation, "cudaErrorMemoryAllocation", "out of memory"},
    {cudaErrorInitializationError, "cudaErrorInitializationError", "initialization error"},
    {cudaErrorInvalidConfiguration, "cudaErrorInvalidConfiguration", "invalid configuration argument"},
    {cudaErrorInvalidSymbol, "cudaErrorInvalidSymbol", "invalid device symbol"},
    {cudaErrorInvalidDevicePointer, "cudaErrorInvalidDevicePointer", "invalid device pointer"},
    {cudaErrorInvalidMemcpyDirection, "cudaErrorInvalidMemcpyDirection", "invalid copy direction for memcpy"},
    {cudaErrorMissingConfiguration, "cudaErrorMissingConfiguration", "__global__ function call is not configured"},
    {cudaErrorInvalidDeviceFunction, "cudaErrorInvalidDeviceFunction", "invalid device function"},
    {cudaErrorNoDevice, "cudaErrorNoDevice", "no CUDA-capable device is detected"},
    {cudaErrorInvalidDevice, "cudaErrorInvalidDevice", "invalid device ordinal"},
    {cudaErrorInvalidKernelImage, "cudaErrorInvalidKernelImage", "device kernel image is invalid"},
    {cudaErrorNoKernelImageForDevice, "cudaErrorNoKernelImageForDevice", "no kernel image is available for execution on the device"},
    {cudaErrorInvalidPtx, "cudaErrorInvalidPtx", "a PTX JIT compilation failed"},
    {cudaErrorInvalidResourceHandle, "cudaErrorInvalidResourceHandle", "invalid resource handle"},
    {cudaErrorSymbolNotFound, "cudaErrorSymbolNotFound", "named symbol not found"},
    {cudaErrorNotReady, "cudaErrorNotReady", "device not ready"},
    {cudaErrorIllegalAddress, "cudaErrorIllegalAddress", "an illegal memory access was encountered"},
    {cudaErrorLaunchOutOfResources, "cudaErrorLaunchOutOfResources", "too many resources requested for launch"},
    {cudaErrorLaunchTimeout, "cudaErrorLaunchTimeout", "the launch timed out and was terminated"},
    {cudaErrorLaunchFailure, "cudaErrorLaunchFailure", "unspecified launch failure"},
    {cudaErrorNotSupported, "cudaErrorNotSupported", "operation not supported"},
    {cudaErrorNotPermitted, "cudaErrorNotPermitted", "operation not permitted"},
    {cudaErrorUnknown, "cudaErrorUnknown", "unknown error"},
};
MC_API const char *cudaGetErrorString(cudaError_t e) {
    for (size_t i = 0; i < sizeof RT_ERRS / sizeof RT_ERRS[0]; i++) if (RT_ERRS[i].e == e) return RT_ERRS[i].str;
    return "unrecognized error code";
}
MC_API const char *cudaGetErrorName(cudaError_t e) {
    for (size_t i = 0; i < sizeof RT_ERRS / sizeof RT_ERRS[0]; i++) if (RT_ERRS[i].e == e) return RT_ERRS[i].name;
    return "cudaErrorUnknown";
}
MC_API cudaError_t cudaGetLastError(void) { cudaError_t e = t_last_error; t_last_error = cudaSuccess; return e; }
MC_API cudaError_t cudaPeekAtLastError(void) { return t_last_error; }

/* ───────────────────────────── device ───────────────────────────── */

MC_API cudaError_t cudaDriverGetVersion(int *v) { RET(cuDriverGetVersion(v)); }
MC_API cudaError_t cudaRuntimeGetVersion(int *v) { if (!v) return ret(cudaErrorInvalidValue); *v = CUDART_VERSION; return cudaSuccess; }
MC_API cudaError_t cudaGetDeviceCount(int *count) {
    CUresult r = cuInit(0);
    if (r) { if (count) *count = 0; RET(r); }
    RET(cuDeviceGetCount(count));
}
static __thread int t_device = 0;
MC_API cudaError_t cudaGetDevice(int *d) { if (!d) return ret(cudaErrorInvalidValue); *d = t_device; return cudaSuccess; }
MC_API cudaError_t cudaSetDevice(int d) {
    CUresult r = cuInit(0);
    if (r) RET(r);
    if (d != 0) return ret(cudaErrorInvalidDevice);
    t_device = d;
    return cudaSuccess;
}
MC_API cudaError_t cudaSetDeviceFlags(unsigned int flags) { (void)flags; return cudaSuccess; }
MC_API cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
    CUresult r = cuInit(0);
    if (r) RET(r);
    RET(cuDeviceGetAttribute(value, (CUdevice_attribute)attr, device));
}
MC_API cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *p, int device) {
    CUresult r = cuInit(0);
    if (r) RET(r);
    if (!p) return ret(cudaErrorInvalidValue);
    if (device != 0) return ret(cudaErrorInvalidDevice);
    memset(p, 0, sizeof *p);
    snprintf(p->name, sizeof p->name, "%s", mc_device_name());
    cuDeviceGetUuid((CUuuid *)&p->uuid, 0);
    int v;
#define A(attr) (cuDeviceGetAttribute(&v, attr, 0), v)
    p->totalGlobalMem = mc_total_mem();
    p->sharedMemPerBlock = (size_t)A(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    p->regsPerBlock = A(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
    p->warpSize = 32;
    p->memPitch = (size_t)A(CU_DEVICE_ATTRIBUTE_MAX_PITCH);
    p->maxThreadsPerBlock = A(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    p->maxThreadsDim[0] = A(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    p->maxThreadsDim[1] = A(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
    p->maxThreadsDim[2] = A(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
    p->maxGridSize[0] = A(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    p->maxGridSize[1] = A(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
    p->maxGridSize[2] = A(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
    p->clockRate = A(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
    p->totalConstMem = (size_t)A(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
    p->major = A(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    p->minor = A(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    p->textureAlignment = 256;
    p->texturePitchAlignment = 256;
    p->deviceOverlap = 0;     /* copies are CPU memcpys after a GPU sync: no copy/compute overlap */
    p->multiProcessorCount = mc_gpu_cores();
    p->kernelExecTimeoutEnabled = 1;
    p->integrated = 1;
    p->canMapHostMemory = 1;
    p->concurrentKernels = 0; /* all streams share one in-order Metal queue */
    p->asyncEngineCount = 0;
    p->unifiedAddressing = 1;
    p->memoryClockRate = A(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
    p->memoryBusWidth = A(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
    p->l2CacheSize = A(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
    p->maxThreadsPerMultiProcessor = A(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
    p->globalL1CacheSupported = 1;
    p->localL1CacheSupported = 1;
    p->sharedMemPerMultiprocessor = (size_t)A(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
    p->regsPerMultiprocessor = 65536;
    p->managedMemory = 1;
    p->singleToDoublePrecisionPerfRatio = 64;
    p->concurrentManagedAccess = 1;
    p->computePreemptionSupported = 1;
    p->sharedMemPerBlockOptin = p->sharedMemPerBlock;
    p->directManagedMemAccessFromHost = 1;
    p->maxBlocksPerMultiProcessor = 32;
    p->reservedSharedMemPerBlock = 32;
    p->hostRegisterSupported = 1;
#undef A
    return cudaSuccess;
}
MC_API cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *p, int device) { return cudaGetDeviceProperties_v2(p, device); }
MC_API cudaError_t cudaDeviceSynchronize(void) { RET(cuCtxSynchronize()); }
MC_API cudaError_t cudaThreadSynchronize(void) { return cudaDeviceSynchronize(); }
MC_API cudaError_t cudaDeviceReset(void) { RET(cuCtxSynchronize()); }
MC_API cudaError_t cudaDeviceSetLimit(enum cudaLimit l, size_t v) { RET(cuCtxSetLimit((CUlimit)l, v)); }
MC_API cudaError_t cudaDeviceGetLimit(size_t *v, enum cudaLimit l) { RET(cuCtxGetLimit(v, (CUlimit)l)); }
MC_API cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache c) { (void)c; return cudaSuccess; }
MC_API cudaError_t cudaDeviceGetStreamPriorityRange(int *lo, int *hi) { if (lo) *lo = 0; if (hi) *hi = 0; return cudaSuccess; }

/* ───────────────────────────── memory ───────────────────────────── */

MC_API cudaError_t cudaMalloc(void **p, size_t size) {
    if (!p) return ret(cudaErrorInvalidValue);
    if (size == 0) { *p = NULL; return cudaSuccess; }
    CUdeviceptr d = 0;
    CUresult r = cuMemAlloc_v2(&d, size);
    *p = (void *)(uintptr_t)d;
    RET(r);
}
MC_API cudaError_t cudaMallocManaged(void **p, size_t size, unsigned int flags) {
    if (!p) return ret(cudaErrorInvalidValue);
    CUdeviceptr d = 0;
    CUresult r = cuMemAllocManaged(&d, size, flags);
    *p = (void *)(uintptr_t)d;
    RET(r);
}
MC_API cudaError_t cudaMallocHost(void **p, size_t size) { RET(cuMemHostAlloc(p, size, 0)); }
MC_API cudaError_t cudaHostAlloc(void **p, size_t size, unsigned int flags) { RET(cuMemHostAlloc(p, size, flags)); }
MC_API cudaError_t cudaMallocPitch(void **p, size_t *pitch, size_t w, size_t h) {
    CUdeviceptr d = 0;
    CUresult r = cuMemAllocPitch_v2(&d, pitch, w, h, 4);
    if (p) *p = (void *)(uintptr_t)d;
    RET(r);
}
MC_API cudaError_t cudaFree(void *p) { RET(cuMemFree_v2((CUdeviceptr)(uintptr_t)p)); }
MC_API cudaError_t cudaFreeHost(void *p) { RET(cuMemFreeHost(p)); }
MC_API cudaError_t cudaHostGetDevicePointer(void **pd, void *ph, unsigned int flags) {
    CUdeviceptr d = 0;
    CUresult r = cuMemHostGetDevicePointer_v2(&d, ph, flags);
    if (pd) *pd = (void *)(uintptr_t)d;
    RET(r);
}
MC_API cudaError_t cudaHostRegister(void *p, size_t n, unsigned int f) { RET(cuMemHostRegister_v2(p, n, f)); }
MC_API cudaError_t cudaHostUnregister(void *p) { RET(cuMemHostUnregister(p)); }
MC_API cudaError_t cudaMemGetInfo(size_t *f, size_t *t) { RET(cuMemGetInfo_v2(f, t)); }
MC_API cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *a, const void *ptr) {
    if (!a) return ret(cudaErrorInvalidValue);
    int is_dev = 0, is_man = 0;
    void *host = NULL;
    uint64_t dev = 0;
    memset(a, 0, sizeof *a);
    if (!mc_pointer_info(ptr, &is_dev, &is_man, &host, &dev)) {
        a->type = cudaMemoryTypeUnregistered;
        a->device = -2;
        a->hostPointer = (void *)ptr;
        return cudaSuccess;
    }
    a->type = is_man ? cudaMemoryTypeManaged : is_dev ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
    a->device = 0;
    a->devicePointer = (void *)(uintptr_t)dev;
    a->hostPointer = is_dev ? NULL : host;
    return cudaSuccess;
}

MC_API cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, enum cudaMemcpyKind kind) {
    if ((unsigned)kind > cudaMemcpyDefault) return ret(cudaErrorInvalidMemcpyDirection);
    if (kind == cudaMemcpyHostToHost) { CUresult r = mc_sync_all(); if (r) RET(r); memmove(dst, src, n); return cudaSuccess; }
    if (kind == cudaMemcpyHostToDevice) RET(cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)dst, src, n));
    if (kind == cudaMemcpyDeviceToHost) RET(cuMemcpyDtoH_v2(dst, (CUdeviceptr)(uintptr_t)src, n));
    if (kind == cudaMemcpyDeviceToDevice) RET(cuMemcpyDtoD_v2((CUdeviceptr)(uintptr_t)dst, (CUdeviceptr)(uintptr_t)src, n));
    RET(cuMemcpy((CUdeviceptr)(uintptr_t)dst, (CUdeviceptr)(uintptr_t)src, n));
}
MC_API cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, enum cudaMemcpyKind kind, cudaStream_t s) {
    (void)s;
    return cudaMemcpy(dst, src, n, kind);
}
MC_API cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t w, size_t h, enum cudaMemcpyKind kind) {
    for (size_t r = 0; r < h; r++) {
        cudaError_t e = cudaMemcpy((char *)dst + r * dpitch, (const char *)src + r * spitch, w, kind);
        if (e) return e;
    }
    return cudaSuccess;
}
MC_API cudaError_t cudaMemset(void *p, int v, size_t n) { RET(cuMemsetD8_v2((CUdeviceptr)(uintptr_t)p, (unsigned char)v, n)); }
MC_API cudaError_t cudaMemsetAsync(void *p, int v, size_t n, cudaStream_t s) { (void)s; return cudaMemset(p, v, n); }
MC_API cudaError_t cudaMemPrefetchAsync(const void *p, size_t n, int d, cudaStream_t s) { (void)p; (void)n; (void)d; (void)s; return cudaSuccess; }
MC_API cudaError_t cudaMemAdvise(const void *p, size_t n, int a, int d) { (void)p; (void)n; (void)a; (void)d; return cudaSuccess; }

/* ───────────────────────────── streams & events ───────────────────────────── */

MC_API cudaError_t cudaStreamCreate(cudaStream_t *s) { RET(cuStreamCreate(s, 0)); }
MC_API cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned int f) { RET(cuStreamCreate(s, f)); }
MC_API cudaError_t cudaStreamCreateWithPriority(cudaStream_t *s, unsigned int f, int p) { RET(cuStreamCreateWithPriority(s, f, p)); }
MC_API cudaError_t cudaStreamDestroy(cudaStream_t s) { RET(cuStreamDestroy_v2(s)); }
MC_API cudaError_t cudaStreamSynchronize(cudaStream_t s) { RET(cuStreamSynchronize(s)); }
MC_API cudaError_t cudaStreamQuery(cudaStream_t s) { CUresult r = cuStreamQuery(s); return r == CUDA_ERROR_NOT_READY ? cudaErrorNotReady : ret(map_err(r)); }
MC_API cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int f) { RET(cuStreamWaitEvent(s, e, f)); }
MC_API cudaError_t cudaEventCreate(cudaEvent_t *e) { RET(cuEventCreate(e, 0)); }
MC_API cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned int f) { RET(cuEventCreate(e, f)); }
MC_API cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) { RET(cuEventRecord(e, s)); }
MC_API cudaError_t cudaEventSynchronize(cudaEvent_t e) { RET(cuEventSynchronize(e)); }
MC_API cudaError_t cudaEventQuery(cudaEvent_t e) { CUresult r = cuEventQuery(e); return r == CUDA_ERROR_NOT_READY ? cudaErrorNotReady : ret(map_err(r)); }
MC_API cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) { RET(cuEventElapsedTime(ms, a, b)); }
MC_API cudaError_t cudaEventDestroy(cudaEvent_t e) { RET(cuEventDestroy_v2(e)); }

/* ───────────────────────────── registration ABI ───────────────────────────── */

typedef struct fatbin_reg {
    const void *wrapper;
    CUmodule mod;
    int load_state; /* 0 = not loaded, 1 = ok, -1 = failed */
    CUresult load_err;
} fatbin_reg_t;

typedef struct {
    const void *host;
    fatbin_reg_t *fb;
    char *name;
    CUfunction f;
} func_reg_t;

typedef struct {
    const void *host;
    fatbin_reg_t *fb;
    char *name;
    size_t size;
    CUdeviceptr dptr;
    void **managed_ptr; /* __managed__ variables: host-side pointer slot */
} var_reg_t;

static func_reg_t *g_funcs;
static int g_nfuncs, g_capfuncs;
static var_reg_t *g_vars;
static int g_nvars, g_capvars;

static CUresult ensure_loaded(fatbin_reg_t *fb) {
    if (fb->load_state == 1) return CUDA_SUCCESS;
    if (fb->load_state == -1) return fb->load_err;
    CUresult r = cuInit(0);
    if (!r) r = cuModuleLoadFatBinary(&fb->mod, fb->wrapper);
    fb->load_state = r ? -1 : 1;
    fb->load_err = r;
    return r;
}

MC_API void **__cudaRegisterFatBinary(void *fatCubin) {
    fatbin_reg_t *fb = (fatbin_reg_t *)calloc(1, sizeof *fb);
    fb->wrapper = fatCubin;
    mc_log(2, "__cudaRegisterFatBinary(%p)", fatCubin);
    return (void **)fb;
}
MC_API void __cudaRegisterFatBinaryEnd(void **handle) { (void)handle; }
MC_API void __cudaUnregisterFatBinary(void **handle) { (void)handle; /* modules live until process exit */ }
MC_API char __cudaInitModule(void **handle) { return ensure_loaded((fatbin_reg_t *)handle) == CUDA_SUCCESS; }

MC_API void __cudaRegisterFunction(void **handle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit,
                                   uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    (void)deviceFun; (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;
    pthread_mutex_lock(&g_rt_mtx);
    if (g_nfuncs == g_capfuncs) {
        g_capfuncs = g_capfuncs ? g_capfuncs * 2 : 32;
        g_funcs = (func_reg_t *)realloc(g_funcs, (size_t)g_capfuncs * sizeof *g_funcs);
    }
    func_reg_t *f = &g_funcs[g_nfuncs++];
    f->host = hostFun;
    f->fb = (fatbin_reg_t *)handle;
    f->name = strdup(deviceName);
    f->f = NULL;
    pthread_mutex_unlock(&g_rt_mtx);
    mc_log(2, "__cudaRegisterFunction(%p, \"%s\")", (const void *)hostFun, deviceName);
}

static void add_var(void **handle, const void *host, const char *name, size_t size, void **managed) {
    pthread_mutex_lock(&g_rt_mtx);
    if (g_nvars == g_capvars) {
        g_capvars = g_capvars ? g_capvars * 2 : 32;
        g_vars = (var_reg_t *)realloc(g_vars, (size_t)g_capvars * sizeof *g_vars);
    }
    var_reg_t *v = &g_vars[g_nvars++];
    memset(v, 0, sizeof *v);
    v->host = host;
    v->fb = (fatbin_reg_t *)handle;
    v->name = strdup(name);
    v->size = size;
    v->managed_ptr = managed;
    pthread_mutex_unlock(&g_rt_mtx);
}

MC_API void __cudaRegisterVar(void **handle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant,
                              int global) {
    (void)deviceAddress; (void)ext; (void)constant; (void)global;
    add_var(handle, hostVar, deviceName, size, NULL);
}

MC_API void __cudaRegisterManagedVar(void **handle, void **hostVarPtrAddress, char *deviceAddress, const char *deviceName, int ext, size_t size,
                                     int constant, int global) {
    (void)deviceAddress; (void)ext; (void)constant; (void)global;
    add_var(handle, hostVarPtrAddress, deviceName, size, hostVarPtrAddress);
    /* Managed variables must be host-addressable right away: the module's
     * globals live in unified memory, so point the host slot at them. */
    var_reg_t *v = &g_vars[g_nvars - 1];
    if (ensure_loaded(v->fb) == CUDA_SUCCESS && cuModuleGetGlobal_v2(&v->dptr, NULL, v->fb->mod, v->name) == CUDA_SUCCESS) {
        void *host = NULL;
        if (mc_pointer_info((void *)(uintptr_t)v->dptr, NULL, NULL, &host, NULL) && hostVarPtrAddress) *hostVarPtrAddress = host;
    }
}

MC_API void __cudaRegisterTexture(void **h, const void *hostVar, const void **devAddr, const char *name, int dim, int norm, int ext) {
    (void)h; (void)hostVar; (void)devAddr; (void)name; (void)dim; (void)norm; (void)ext;
}
MC_API void __cudaRegisterSurface(void **h, const void *hostVar, const void **devAddr, const char *name, int dim, int ext) {
    (void)h; (void)hostVar; (void)devAddr; (void)name; (void)dim; (void)ext;
}

static func_reg_t *lookup_func(const void *host) {
    for (int i = 0; i < g_nfuncs; i++) if (g_funcs[i].host == host) return &g_funcs[i];
    return NULL;
}
static var_reg_t *lookup_var(const void *host) {
    for (int i = 0; i < g_nvars; i++) if (g_vars[i].host == host) return &g_vars[i];
    return NULL;
}

static CUresult resolve_func(const void *host, CUfunction *out) {
    pthread_mutex_lock(&g_rt_mtx);
    func_reg_t *f = lookup_func(host);
    CUresult r = CUDA_SUCCESS;
    if (!f) r = CUDA_ERROR_INVALID_HANDLE;
    else if (!f->f) {
        r = ensure_loaded(f->fb);
        if (!r) r = cuModuleGetFunction(&f->f, f->fb->mod, f->name);
    }
    if (!r) *out = f->f;
    pthread_mutex_unlock(&g_rt_mtx);
    return r;
}

static CUresult resolve_var(const void *host, CUdeviceptr *dptr, size_t *size) {
    pthread_mutex_lock(&g_rt_mtx);
    var_reg_t *v = lookup_var(host);
    CUresult r = CUDA_SUCCESS;
    if (!v) r = CUDA_ERROR_NOT_FOUND;
    else if (!v->dptr) {
        r = ensure_loaded(v->fb);
        if (!r) r = cuModuleGetGlobal_v2(&v->dptr, &v->size, v->fb->mod, v->name);
    }
    if (!r) { *dptr = v->dptr; if (size) *size = v->size; }
    pthread_mutex_unlock(&g_rt_mtx);
    return r;
}

/* ───────────────────────────── launch ───────────────────────────── */

MC_API cudaError_t cudaLaunchKernel(const void *func, dim3 g, dim3 b, void **args, size_t shmem, cudaStream_t stream) {
    CUfunction f = NULL;
    CUresult r = resolve_func(func, &f);
    if (r) {
        if (r == CUDA_ERROR_INVALID_HANDLE) return ret(cudaErrorInvalidDeviceFunction);
        RET(r);
    }
    r = cuLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, (unsigned)shmem, stream, args, NULL);
    if (r == CUDA_ERROR_INVALID_VALUE) return ret(cudaErrorInvalidConfiguration);
    RET(r);
}
MC_API cudaError_t cudaLaunchKernel_ptsz(const void *func, dim3 g, dim3 b, void **args, size_t shmem, cudaStream_t stream) {
    return cudaLaunchKernel(func, g, b, args, shmem, stream);
}
MC_API cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 g, dim3 b, void **args, size_t shmem, cudaStream_t stream) {
    return cudaLaunchKernel(func, g, b, args, shmem, stream);
}

typedef struct { dim3 grid, block; size_t shmem; void *stream; } call_cfg_t;
static __thread call_cfg_t t_cfg_stack[32];
static __thread int t_cfg_depth;

MC_API unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shmem, struct CUstream_st *stream) {
    if (t_cfg_depth >= 32) return 1;
    call_cfg_t *c = &t_cfg_stack[t_cfg_depth++];
    c->grid = grid;
    c->block = block;
    c->shmem = shmem;
    c->stream = stream;
    return 0;
}
MC_API cudaError_t __cudaPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shmem, void *stream) {
    if (t_cfg_depth <= 0) return cudaErrorMissingConfiguration;
    call_cfg_t *c = &t_cfg_stack[--t_cfg_depth];
    if (grid) *grid = c->grid;
    if (block) *block = c->block;
    if (shmem) *shmem = c->shmem;
    if (stream) *(void **)stream = c->stream;
    return cudaSuccess;
}

/* legacy launch protocol (cudaConfigureCall + cudaSetupArgument + cudaLaunch) */
static __thread call_cfg_t t_legacy_cfg;
static __thread unsigned char t_legacy_args[4096];
static __thread size_t t_legacy_size;
MC_API cudaError_t cudaConfigureCall(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
    t_legacy_cfg.grid = grid;
    t_legacy_cfg.block = block;
    t_legacy_cfg.shmem = shmem;
    t_legacy_cfg.stream = stream;
    t_legacy_size = 0;
    return cudaSuccess;
}
MC_API cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
    if (offset + size > sizeof t_legacy_args) return ret(cudaErrorInvalidValue);
    memcpy(t_legacy_args + offset, arg, size);
    if (offset + size > t_legacy_size) t_legacy_size = offset + size;
    return cudaSuccess;
}
MC_API cudaError_t cudaLaunch(const void *func) {
    CUfunction f = NULL;
    CUresult r = resolve_func(func, &f);
    if (r) return ret(cudaErrorInvalidDeviceFunction);
    size_t sz = t_legacy_size;
    void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, t_legacy_args, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
    call_cfg_t *c = &t_legacy_cfg;
    r = cuLaunchKernel(f, c->grid.x, c->grid.y, c->grid.z, c->block.x, c->block.y, c->block.z, (unsigned)c->shmem, (CUstream)c->stream, NULL, extra);
    if (r == CUDA_ERROR_INVALID_VALUE) return ret(cudaErrorInvalidConfiguration);
    RET(r);
}

MC_API cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *a, const void *func) {
    if (!a) return ret(cudaErrorInvalidValue);
    CUfunction f = NULL;
    CUresult r = resolve_func(func, &f);
    if (r) return ret(cudaErrorInvalidDeviceFunction);
    memset(a, 0, sizeof *a);
    int mt = 0, ss = 0, lb = 0, dyn = 0;
    mc_function_info(f, &mt, &ss, &lb);
    cuFuncGetAttribute(&dyn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f);
    a->maxThreadsPerBlock = mt;
    a->sharedSizeBytes = (size_t)ss;
    a->localSizeBytes = (size_t)lb;
    a->numRegs = 32;
    a->ptxVersion = 80;
    a->binaryVersion = 80;
    a->maxDynamicSharedSizeBytes = dyn;
    a->preferredShmemCarveout = -1;
    return cudaSuccess;
}
MC_API cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
    CUfunction f = NULL;
    CUresult r = resolve_func(func, &f);
    if (r) return ret(cudaErrorInvalidDeviceFunction);
    RET(cuFuncSetAttribute(f, (CUfunction_attribute)attr, value));
}
MC_API cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache c) { (void)func; (void)c; return cudaSuccess; }
MC_API cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *n, const void *func, int bs, size_t dsm) {
    CUfunction f = NULL;
    CUresult r = resolve_func(func, &f);
    if (r) return ret(cudaErrorInvalidDeviceFunction);
    RET(cuOccupancyMaxActiveBlocksPerMultiprocessor(n, f, bs, dsm));
}

/* ───────────────────────────── symbols ───────────────────────────── */

MC_API cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    CUdeviceptr d = 0;
    CUresult r = resolve_var(symbol, &d, NULL);
    if (r) return ret(cudaErrorInvalidSymbol);
    if (devPtr) *devPtr = (void *)(uintptr_t)d;
    return cudaSuccess;
}
MC_API cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
    CUdeviceptr d = 0;
    CUresult r = resolve_var(symbol, &d, size);
    return r ? ret(cudaErrorInvalidSymbol) : cudaSuccess;
}
MC_API cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t n, size_t offset, enum cudaMemcpyKind kind) {
    CUdeviceptr d = 0;
    size_t sz = 0;
    if (resolve_var(symbol, &d, &sz)) return ret(cudaErrorInvalidSymbol);
    if (offset + n > sz) return ret(cudaErrorInvalidValue);
    return cudaMemcpy((void *)(uintptr_t)(d + offset), src, n, kind == cudaMemcpyDeviceToDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyDefault);
}
MC_API cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t n, size_t offset, enum cudaMemcpyKind kind) {
    CUdeviceptr d = 0;
    size_t sz = 0;
    (void)kind;
    if (resolve_var(symbol, &d, &sz)) return ret(cudaErrorInvalidSymbol);
    if (offset + n > sz) return ret(cudaErrorInvalidValue);
    return cudaMemcpy(dst, (const void *)(uintptr_t)(d + offset), n, cudaMemcpyDefault);
}
MC_API cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t n, size_t offset, enum cudaMemcpyKind kind, cudaStream_t s) {
    (void)s;
    return cudaMemcpyToSymbol(symbol, src, n, offset, kind);
}
