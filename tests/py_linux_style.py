"""A Linux-style CUDA Driver API script — it knows nothing about MetaCuda.

It hard-codes `libcuda.so.1` like numba/cupy/pycuda-era code does. On macOS this
only works when libmetalcuda.dylib is injected (python -m metalcuda ...), whose
dyld interposer redirects the dlopen.

usage: py_linux_style.py <basic.ptx>
"""
import ctypes
import sys

import numpy as np

cuda = ctypes.CDLL("libcuda.so.1")


def check(r, what):
    if r != 0:
        raise SystemExit(f"{what} failed with CUresult {r}")


check(cuda.cuInit(0), "cuInit")
dev = ctypes.c_int()
check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
name = ctypes.create_string_buffer(128)
check(cuda.cuDeviceGetName(name, 128, dev), "cuDeviceGetName")
ctx = ctypes.c_void_p()
check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

ptx = open(sys.argv[1], "rb").read() + b"\0"
mod = ctypes.c_void_p()
check(cuda.cuModuleLoadData(ctypes.byref(mod), ptx), "cuModuleLoadData")
fn = ctypes.c_void_p()
check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"saxpy"), "cuModuleGetFunction")

n = 1 << 20
x = np.random.default_rng(0).standard_normal(n).astype(np.float32)
y = np.random.default_rng(1).standard_normal(n).astype(np.float32)
dx, dy = ctypes.c_uint64(), ctypes.c_uint64()
check(cuda.cuMemAlloc_v2(ctypes.byref(dx), ctypes.c_size_t(x.nbytes)), "cuMemAlloc")
check(cuda.cuMemAlloc_v2(ctypes.byref(dy), ctypes.c_size_t(y.nbytes)), "cuMemAlloc")
check(cuda.cuMemcpyHtoD_v2(dx, x.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(x.nbytes)), "HtoD")
check(cuda.cuMemcpyHtoD_v2(dy, y.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(y.nbytes)), "HtoD")

cn, alpha = ctypes.c_int(n), ctypes.c_float(3.0)
args = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in (cn, alpha, dx, dy)])
cuda.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
check(cuda.cuLaunchKernel(fn, (n + 255) // 256, 1, 1, 256, 1, 1, 0, None, args, None), "cuLaunchKernel")
check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
out = np.empty_like(y)
check(cuda.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), dy, ctypes.c_size_t(out.nbytes)), "DtoH")
err = float(np.max(np.abs(out - (3.0 * x + y))))
print(f"linux-style script on '{name.value.decode()}': saxpy n={n} max error {err:.2e}")
sys.exit(0 if err < 1e-5 else 1)
