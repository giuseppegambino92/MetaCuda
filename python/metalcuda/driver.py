"""Minimal NumPy-friendly wrapper over the CUDA Driver API (works with any
libcuda — MetaCuda or NVIDIA's).

    from metalcuda.driver import Context
    ctx = Context()                       # cuInit + primary context
    mod = ctx.load_ptx(open("k.ptx").read())
    k = mod.function("vecadd")
    a = ctx.to_device(np.arange(10, dtype=np.float32)); ...
    k((grid,), (block,), a, b, c, np.int32(n))
    c.to_host()
"""
from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np

CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_uint64


class CudaError(RuntimeError):
    def __init__(self, lib, code, what):
        name = ctypes.c_char_p()
        lib.cuGetErrorName(code, ctypes.byref(name))
        super().__init__(f"{what} failed: {name.value.decode() if name.value else code}")
        self.code = code


def _load_libcuda(path=None):
    if path:
        return ctypes.CDLL(path)
    for name in ("libcuda.so.1", "libcuda.dylib", ctypes.util.find_library("cuda")):
        if name:
            try:
                return ctypes.CDLL(name)
            except OSError:
                pass
    raise OSError("no CUDA driver library found (call metalcuda.enable() or run under `python -m metalcuda`)")


class DeviceArray:
    def __init__(self, ctx, shape, dtype):
        self.ctx, self.shape, self.dtype = ctx, tuple(shape), np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.ptr = CUdeviceptr()
        ctx._check(ctx.lib.cuMemAlloc_v2(ctypes.byref(self.ptr), ctypes.c_size_t(max(self.nbytes, 1))), "cuMemAlloc")

    def upload(self, host: np.ndarray):
        host = np.ascontiguousarray(host, dtype=self.dtype)
        self.ctx._check(self.ctx.lib.cuMemcpyHtoD_v2(self.ptr, host.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(host.nbytes)), "cuMemcpyHtoD")
        return self

    def to_host(self) -> np.ndarray:
        out = np.empty(self.shape, self.dtype)
        self.ctx._check(self.ctx.lib.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), self.ptr, ctypes.c_size_t(self.nbytes)), "cuMemcpyDtoH")
        return out

    def zero(self):
        self.ctx._check(self.ctx.lib.cuMemsetD8_v2(self.ptr, ctypes.c_ubyte(0), ctypes.c_size_t(self.nbytes)), "cuMemsetD8")
        return self

    def free(self):
        if self.ptr.value:
            self.ctx.lib.cuMemFree_v2(self.ptr)
            self.ptr = CUdeviceptr(0)

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class Function:
    def __init__(self, module, name):
        self.module, self.name = module, name
        self.handle = ctypes.c_void_p()
        module.ctx._check(module.ctx.lib.cuModuleGetFunction(ctypes.byref(self.handle), module.handle, name.encode()), f"cuModuleGetFunction({name})")

    def __call__(self, grid, block, *args, shared_mem=0, stream=None):
        grid = tuple(grid) + (1,) * (3 - len(grid))
        block = tuple(block) + (1,) * (3 - len(block))
        holders = []
        for a in args:
            if isinstance(a, DeviceArray):
                holders.append(CUdeviceptr(a.ptr.value))
            elif isinstance(a, np.generic):
                holders.append(np.ctypeslib.as_ctypes(np.array(a)))
            elif isinstance(a, ctypes._SimpleCData) or isinstance(a, ctypes.Structure):
                holders.append(a)
            else:
                raise TypeError(f"kernel argument of type {type(a).__name__}: pass numpy scalars (np.int32, np.float32, ...) or DeviceArrays")
        params = (ctypes.c_void_p * max(len(holders), 1))(*[ctypes.cast(ctypes.pointer(h), ctypes.c_void_p) for h in holders])
        lib = self.module.ctx.lib
        self.module.ctx._check(
            lib.cuLaunchKernel(self.handle, *[ctypes.c_uint(v) for v in grid], *[ctypes.c_uint(v) for v in block], ctypes.c_uint(shared_mem),
                               stream, params, None), f"cuLaunchKernel({self.name})")


class Module:
    def __init__(self, ctx, image: bytes):
        self.ctx = ctx
        self.handle = ctypes.c_void_p()
        self._image = ctypes.create_string_buffer(image)
        ctx._check(ctx.lib.cuModuleLoadData(ctypes.byref(self.handle), self._image), "cuModuleLoadData")

    def function(self, name) -> Function:
        return Function(self, name)

    def get_global(self, name):
        p, n = CUdeviceptr(), ctypes.c_size_t()
        self.ctx._check(self.ctx.lib.cuModuleGetGlobal_v2(ctypes.byref(p), ctypes.byref(n), self.handle, name.encode()), "cuModuleGetGlobal")
        return p.value, n.value


class Context:
    def __init__(self, lib_path=None, device=0):
        self.lib = _load_libcuda(lib_path)
        self.lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self._check(self.lib.cuInit(0), "cuInit")
        self.device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), device), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        self._check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.device), "cuDevicePrimaryCtxRetain")
        self._check(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    def _check(self, r, what):
        if r != 0:
            raise CudaError(self.lib, r, what)

    @property
    def name(self) -> str:
        buf = ctypes.create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(buf, 256, self.device), "cuDeviceGetName")
        return buf.value.decode()

    def attribute(self, attr: int) -> int:
        v = ctypes.c_int()
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(v), attr, self.device), "cuDeviceGetAttribute")
        return v.value

    def load_ptx(self, ptx) -> Module:
        return Module(self, ptx.encode() if isinstance(ptx, str) else ptx)

    def empty(self, shape, dtype) -> DeviceArray:
        return DeviceArray(self, shape if isinstance(shape, tuple) else (shape,), dtype)

    def to_device(self, host: np.ndarray) -> DeviceArray:
        return DeviceArray(self, host.shape, host.dtype).upload(host)

    def synchronize(self):
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")
