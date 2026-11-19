/*
 * MetaCuda — CUDA Runtime API header (ABI-compatible subset of CUDA 12.x).
 */
#ifndef METACUDA_CUDA_RUNTIME_API_H
#define METACUDA_CUDA_RUNTIME_API_H

#include <stddef.h>
#include <stdint.h>

#define CUDART_VERSION 12040

#include "vector_types.h"

#ifdef __cplusplus
#define MC_DV(v) = v
#else
#define MC_DV(v)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorMissingConfiguration = 52,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchFailure = 719,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorUnknown = 999
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

typedef enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
} cudaMemoryType;

typedef enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrManagedMemory = 83
} cudaDeviceAttr;

typedef enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9
} cudaFuncAttribute;

typedef enum cudaLimit { cudaLimitStackSize = 0, cudaLimitPrintfFifoSize = 1, cudaLimitMallocHeapSize = 2 } cudaLimit;
typedef enum cudaFuncCache { cudaFuncCachePreferNone = 0, cudaFuncCachePreferShared = 1, cudaFuncCachePreferL1 = 2, cudaFuncCachePreferEqual = 3 } cudaFuncCache;

typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;

#define cudaStreamDefault 0x00
#define cudaStreamNonBlocking 0x01
#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamPerThread ((cudaStream_t)0x2)
#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaHostAllocDefault 0x00
#define cudaHostAllocPortable 0x01
#define cudaHostAllocMapped 0x02
#define cudaHostAllocWriteCombined 0x04
#define cudaHostRegisterDefault 0x00
#define cudaMemAttachGlobal 0x01
#define cudaMemAttachHost 0x02
#define cudaCpuDeviceId ((int)-1)

#ifndef METACUDA_CUUUID_DEFINED
#define METACUDA_CUUUID_DEFINED
struct CUuuid_st { char bytes[16]; };
#endif
typedef struct CUuuid_st cudaUUID_t;

/* Layout of cudaDeviceProp as of CUDA 12.x (cudaGetDeviceProperties_v2). */
struct cudaDeviceProp {
    char name[256];
    cudaUUID_t uuid;
    char luid[8];
    unsigned int luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
    int hostRegisterSupported;
    int sparseCudaArraySupported;
    int hostRegisterReadOnlySupported;
    int timelineSemaphoreInteropSupported;
    int memoryPoolsSupported;
    int gpuDirectRDMASupported;
    unsigned int gpuDirectRDMAFlushWritesOptions;
    int gpuDirectRDMAWritesOrdering;
    unsigned int memoryPoolSupportedHandleTypes;
    int deferredMappingCudaArraySupported;
    int ipcEventSupported;
    int clusterLaunch;
    int unifiedFunctionPointers;
    int reserved2[2];
    int reserved1[1];
    int reserved[60];
};

struct cudaFuncAttributes {
    size_t sharedSizeBytes;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int ptxVersion;
    int binaryVersion;
    int cacheModeCA;
    int maxDynamicSharedSizeBytes;
    int preferredShmemCarveout;
    int clusterDimMustBeSet;
    int requiredClusterWidth;
    int requiredClusterHeight;
    int requiredClusterDepth;
    int clusterSchedulingPolicyPreference;
    int nonPortableClusterSizeAllowed;
    int reserved[16];
};

struct cudaPointerAttributes {
    enum cudaMemoryType type;
    int device;
    void *devicePointer;
    void *hostPointer;
};

#ifndef METACUDA_INTERNAL
#define cudaGetDeviceProperties cudaGetDeviceProperties_v2
#endif

const char *cudaGetErrorString(cudaError_t error);
const char *cudaGetErrorName(cudaError_t error);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);

cudaError_t cudaDriverGetVersion(int *driverVersion);
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device);
cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaThreadSynchronize(void);
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
cudaError_t cudaSetDeviceFlags(unsigned int flags);

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags MC_DV(cudaMemAttachGlobal));
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void *ptr);
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr);

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream MC_DV(0));
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream MC_DV(0));
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset MC_DV(0), enum cudaMemcpyKind kind MC_DV(cudaMemcpyHostToDevice));
cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset MC_DV(0), enum cudaMemcpyKind kind MC_DV(cudaMemcpyDeviceToHost));
cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol);
cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol);
cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream MC_DV(0));

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags MC_DV(0));

cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream MC_DV(0));
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy(cudaEvent_t event);

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem MC_DV(0), cudaStream_t stream MC_DV(0));
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);
cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);

/* Launch configuration entry points used by compiler-generated `<<<...>>>` code. */
#ifdef __cplusplus
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, struct CUstream_st *stream = 0);
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
#else
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream);
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
#endif
cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset);
cudaError_t cudaLaunch(const void *func);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
/* C++ conveniences provided by NVIDIA's cuda_runtime.h */
template <class T> static inline cudaError_t cudaMalloc(T **p, size_t s) { return cudaMalloc((void **)(void *)p, s); }
template <class T> static inline cudaError_t cudaMallocManaged(T **p, size_t s, unsigned int f = cudaMemAttachGlobal) { return cudaMallocManaged((void **)(void *)p, s, f); }
template <class T> static inline cudaError_t cudaMallocHost(T **p, size_t s) { return cudaMallocHost((void **)(void *)p, s); }
template <class T> static inline cudaError_t cudaHostAlloc(T **p, size_t s, unsigned int f) { return cudaHostAlloc((void **)(void *)p, s, f); }
template <class T> static inline cudaError_t cudaMallocPitch(T **p, size_t *pitch, size_t w, size_t h) { return cudaMallocPitch((void **)(void *)p, pitch, w, h); }
template <class T> static inline cudaError_t cudaMemcpyToSymbol(const T &symbol, const void *src, size_t count, size_t offset = 0, enum cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
    return cudaMemcpyToSymbol((const void *)&symbol, src, count, offset, kind);
}
template <class T> static inline cudaError_t cudaMemcpyFromSymbol(void *dst, const T &symbol, size_t count, size_t offset = 0, enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
    return cudaMemcpyFromSymbol(dst, (const void *)&symbol, count, offset, kind);
}
template <class T> static inline cudaError_t cudaGetSymbolAddress(void **devPtr, const T &symbol) { return cudaGetSymbolAddress(devPtr, (const void *)&symbol); }
template <class T> static inline cudaError_t cudaGetSymbolSize(size_t *size, const T &symbol) { return cudaGetSymbolSize(size, (const void *)&symbol); }
template <class T> static inline cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *a, T *entry) { return cudaFuncGetAttributes(a, (const void *)entry); }
template <class T> static inline cudaError_t cudaFuncSetAttribute(T *entry, enum cudaFuncAttribute attr, int value) { return cudaFuncSetAttribute((const void *)entry, attr, value); }
template <class T> static inline cudaError_t cudaLaunchKernel(T *func, dim3 g, dim3 b, void **args, size_t shmem = 0, cudaStream_t s = 0) {
    return cudaLaunchKernel((const void *)func, g, b, args, shmem, s);
}
template <class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *n, T func, int bs, size_t dsm) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(n, (const void *)func, bs, dsm);
}
#endif
#endif /* METACUDA_CUDA_RUNTIME_API_H */
