/*
 * MetaCuda — CUDA Driver API header (ABI-compatible subset of CUDA 12.x cuda.h).
 *
 * Function names follow NVIDIA's convention: the public names are macros that
 * map to the versioned (_v2) ABI symbols. libmetalcuda exports both the
 * versioned and the legacy unversioned entry points.
 */
#ifndef METACUDA_CUDA_H
#define METACUDA_CUDA_H

#include <stddef.h>
#include <stdint.h>

#define CUDA_VERSION 12040

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CUDAAPI
#define CUDAAPI
#endif

typedef uint64_t CUdeviceptr_v2;
typedef CUdeviceptr_v2 CUdeviceptr;
typedef int CUdevice_v1;
typedef CUdevice_v1 CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUlib_st *CUlibrary;
typedef struct CUkern_st *CUkernel;
typedef unsigned long long CUmemGenericAllocationHandle_v1;
typedef CUmemGenericAllocationHandle_v1 CUmemGenericAllocationHandle;

#ifndef METACUDA_CUUUID_DEFINED
#define METACUDA_CUUUID_DEFINED
struct CUuuid_st { char bytes[16]; };
#endif
typedef struct CUuuid_st CUuuid;

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999
} CUresult;

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_MAX
} CUdevice_attribute;

typedef enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_MAX
} CUfunction_attribute;

typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13
} CUpointer_attribute;

typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 1,
    CU_MEMORYTYPE_DEVICE = 2,
    CU_MEMORYTYPE_ARRAY = 3,
    CU_MEMORYTYPE_UNIFIED = 4
} CUmemorytype;

typedef enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14
} CUjit_option;

typedef enum CUlimit_enum {
    CU_LIMIT_STACK_SIZE = 0,
    CU_LIMIT_PRINTF_FIFO_SIZE = 1,
    CU_LIMIT_MALLOC_HEAP_SIZE = 2
} CUlimit;

typedef enum CUfunc_cache_enum {
    CU_FUNC_CACHE_PREFER_NONE = 0,
    CU_FUNC_CACHE_PREFER_SHARED = 1,
    CU_FUNC_CACHE_PREFER_L1 = 2,
    CU_FUNC_CACHE_PREFER_EQUAL = 3
} CUfunc_cache;

/* Virtual memory management */
typedef enum CUmemAllocationType_enum { CU_MEM_ALLOCATION_TYPE_INVALID = 0, CU_MEM_ALLOCATION_TYPE_PINNED = 1 } CUmemAllocationType;
typedef enum CUmemLocationType_enum { CU_MEM_LOCATION_TYPE_INVALID = 0, CU_MEM_LOCATION_TYPE_DEVICE = 1 } CUmemLocationType;
typedef enum CUmemAccess_flags_enum {
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0,
    CU_MEM_ACCESS_FLAGS_PROT_READ = 1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
} CUmemAccess_flags;
typedef enum CUmemAllocationGranularity_flags_enum {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
} CUmemAllocationGranularity_flags;
typedef struct CUmemLocation_st { CUmemLocationType type; int id; } CUmemLocation;
typedef struct CUmemAllocationProp_st {
    CUmemAllocationType type;
    int requestedHandleTypes;
    CUmemLocation location;
    void *win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
} CUmemAllocationProp;
typedef struct CUmemAccessDesc_st { CUmemLocation location; CUmemAccess_flags flags; } CUmemAccessDesc;

#define CU_LAUNCH_PARAM_END ((void *)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void *)0x02)

#define CU_STREAM_LEGACY ((CUstream)0x1)
#define CU_STREAM_PER_THREAD ((CUstream)0x2)
#define CU_STREAM_DEFAULT 0x0
#define CU_STREAM_NON_BLOCKING 0x1

#define CU_EVENT_DEFAULT 0x0
#define CU_EVENT_BLOCKING_SYNC 0x1
#define CU_EVENT_DISABLE_TIMING 0x2

#define CU_MEMHOSTALLOC_PORTABLE 0x01
#define CU_MEMHOSTALLOC_DEVICEMAP 0x02
#define CU_MEMHOSTALLOC_WRITECOMBINED 0x04
#define CU_MEM_ATTACH_GLOBAL 0x1

#define CU_GET_PROC_ADDRESS_DEFAULT 0
#define CU_GET_PROC_ADDRESS_LEGACY_STREAM (1 << 0)
#define CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM (1 << 1)

typedef enum CUdriverProcAddressQueryResult_enum {
    CU_GET_PROC_ADDRESS_SUCCESS = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2
} CUdriverProcAddressQueryResult;

/* public names → versioned ABI symbols (same as NVIDIA's cuda.h) */
#ifndef METACUDA_INTERNAL
#define cuDeviceTotalMem cuDeviceTotalMem_v2
#define cuCtxCreate cuCtxCreate_v2
#define cuCtxDestroy cuCtxDestroy_v2
#define cuCtxPushCurrent cuCtxPushCurrent_v2
#define cuCtxPopCurrent cuCtxPopCurrent_v2
#define cuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease_v2
#define cuDevicePrimaryCtxReset cuDevicePrimaryCtxReset_v2
#define cuDevicePrimaryCtxSetFlags cuDevicePrimaryCtxSetFlags_v2
#define cuStreamDestroy cuStreamDestroy_v2
#define cuEventDestroy cuEventDestroy_v2
#define cuMemAlloc cuMemAlloc_v2
#define cuMemAllocPitch cuMemAllocPitch_v2
#define cuMemFree cuMemFree_v2
#define cuMemGetInfo cuMemGetInfo_v2
#define cuMemGetAddressRange cuMemGetAddressRange_v2
#define cuMemAllocHost cuMemAllocHost_v2
#define cuMemHostGetDevicePointer cuMemHostGetDevicePointer_v2
#define cuMemHostRegister cuMemHostRegister_v2
#define cuMemcpyHtoD cuMemcpyHtoD_v2
#define cuMemcpyDtoH cuMemcpyDtoH_v2
#define cuMemcpyDtoD cuMemcpyDtoD_v2
#define cuMemcpyHtoDAsync cuMemcpyHtoDAsync_v2
#define cuMemcpyDtoHAsync cuMemcpyDtoHAsync_v2
#define cuMemcpyDtoDAsync cuMemcpyDtoDAsync_v2
#define cuMemsetD8 cuMemsetD8_v2
#define cuMemsetD16 cuMemsetD16_v2
#define cuMemsetD32 cuMemsetD32_v2
#define cuModuleGetGlobal cuModuleGetGlobal_v2
#endif

CUresult CUDAAPI cuInit(unsigned int Flags);
CUresult CUDAAPI cuDriverGetVersion(int *driverVersion);
CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr);
CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr);
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, uint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);

CUresult CUDAAPI cuDeviceGetCount(int *count);
CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal);
CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
CUresult CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);
CUresult CUDAAPI cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags);

CUresult CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult CUDAAPI cuCtxDestroy_v2(CUcontext ctx);
CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx);
CUresult CUDAAPI cuCtxPushCurrent_v2(CUcontext ctx);
CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext *pctx);
CUresult CUDAAPI cuCtxGetDevice(CUdevice *device);
CUresult CUDAAPI cuCtxSynchronize(void);
CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value);
CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit);

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority);
CUresult CUDAAPI cuStreamDestroy_v2(CUstream hStream);
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream);
CUresult CUDAAPI cuStreamQuery(CUstream hStream);
CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);

CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult CUDAAPI cuEventSynchronize(CUevent hEvent);
CUresult CUDAAPI cuEventQuery(CUevent hEvent);
CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult CUDAAPI cuEventDestroy_v2(CUevent hEvent);

CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult CUDAAPI cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
CUresult CUDAAPI cuMemFree_v2(CUdeviceptr dptr);
CUresult CUDAAPI cuMemAllocHost_v2(void **pp, size_t bytesize);
CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);
CUresult CUDAAPI cuMemFreeHost(void *p);
CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags);
CUresult CUDAAPI cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags);
CUresult CUDAAPI cuMemHostUnregister(void *p);
CUresult CUDAAPI cuMemGetInfo_v2(size_t *free, size_t *total);
CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr);

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N);
CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N);
CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);

CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags);
CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
CUresult CUDAAPI cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);

CUresult CUDAAPI cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname);
CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image);
CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
CUresult CUDAAPI cuModuleUnload(CUmodule hmod);
CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);

CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                           unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                           unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);
CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);

#ifdef __cplusplus
}
#endif
#endif /* METACUDA_CUDA_H */
