"""Python / NumPy verification of MetaCuda.

1. An unmodified Linux-style script (ctypes.CDLL("libcuda.so.1")) fails on
   plain macOS and succeeds under `python -m metalcuda` (dyld interposition).
2. In-process `metalcuda.enable()` redirects ctypes / find_library.
3. NumPy workloads through the driver wrapper, checked against NumPy:
   vector add, SAXPY, tiled SGEMM, reduction, histogram, softmax-style math.
"""
import ctypes
import ctypes.util
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
PTX = os.path.join(ROOT, "build", "ptx")

import metalcuda  # noqa: E402

passed = failed = 0


def case(name):
    def deco(fn):
        global passed, failed
        t0 = time.perf_counter()
        try:
            detail = fn()
            passed += 1
            print(f"  [PASS] {name} ({(time.perf_counter() - t0) * 1e3:.0f} ms){'  ' + detail if detail else ''}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {name}: {e}")
        return fn
    return deco


print(f"MetaCuda Python tests — {sys.executable} (numpy {np.__version__})")


@case("unmodified libcuda.so.1 script fails without MetaCuda")
def _():
    env = {k: v for k, v in os.environ.items() if not k.startswith("DYLD_")}
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "py_linux_style.py"), os.path.join(PTX, "basic.ptx")],
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "libcuda.so.1" in r.stderr, f"expected dlopen failure, got rc={r.returncode}"
    return "OSError: libcuda.so.1 not found (as expected on macOS)"


@case("same script under `python -m metalcuda` (DYLD_INSERT_LIBRARIES + dlopen interposition)")
def _():
    env = {k: v for k, v in os.environ.items() if not k.startswith("DYLD_")}
    env["METALCUDA_LOG"] = "1"
    env["PYTHONPATH"] = os.path.join(ROOT, "python")
    r = subprocess.run([sys.executable, "-m", "metalcuda", os.path.join(ROOT, "tests", "py_linux_style.py"), os.path.join(PTX, "basic.ptx")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    assert 'dlopen("libcuda.so.1")' in r.stderr, "interposer did not report the redirect"
    return r.stdout.strip()


@case("in-process metalcuda.enable(): ctypes.CDLL / find_library redirection")
def _():
    metalcuda.enable()
    lib = ctypes.CDLL("libcudart.so.12")
    assert ctypes.util.find_library("cuda") == metalcuda.library_path()
    n = ctypes.c_int()
    assert lib.cudaGetDeviceCount(ctypes.byref(n)) == 0 and n.value == 1
    return f"libcudart.so.12 -> {os.path.basename(metalcuda.library_path())}, cudaGetDeviceCount = {n.value}"


from metalcuda.driver import Context  # noqa: E402

ctx = Context()
basic = ctx.load_ptx(open(os.path.join(PTX, "basic.ptx")).read())
par = ctx.load_ptx(open(os.path.join(PTX, "parallel.ptx")).read())
rng = np.random.default_rng(42)


@case("device query through the driver wrapper")
def _():
    sms = ctx.attribute(16)
    cc = (ctx.attribute(75), ctx.attribute(76))
    return f"{ctx.name}, sm_{cc[0]}{cc[1]}, {sms} SMs"


@case("vector add (NumPy reference)")
def _():
    n = 3_000_017
    a, b = rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32)
    da, db, dc = ctx.to_device(a), ctx.to_device(b), ctx.empty(n, np.float32)
    basic.function("vecadd")(((n + 255) // 256,), (256,), da, db, dc, np.int32(n))
    np.testing.assert_array_equal(dc.to_host(), a + b)


@case("SAXPY grid-stride")
def _():
    n = 1 << 21
    x, y = rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32)
    dx, dy = ctx.to_device(x), ctx.to_device(y)
    basic.function("saxpy")((128,), (256,), np.int32(n), np.float32(-0.75), dx, dy)
    np.testing.assert_allclose(dy.to_host(), np.float32(-0.75) * x + y, rtol=1e-6, atol=1e-6)


@case("tiled SGEMM vs numpy.matmul")
def _():
    M, N, K = 512, 384, 640
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    dA, dB, dC = ctx.to_device(A), ctx.to_device(B), ctx.empty((M, N), np.float32)
    k = par.function("sgemm_tiled")
    k(((N + 15) // 16, (M + 15) // 16), (16, 16), dA, dB, dC, np.int32(M), np.int32(N), np.int32(K))
    ctx.synchronize()
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        k(((N + 15) // 16, (M + 15) // 16), (16, 16), dA, dB, dC, np.int32(M), np.int32(N), np.int32(K))
    ctx.synchronize()
    dt = (time.perf_counter() - t0) / reps
    C = dC.to_host()
    ref = A.astype(np.float64) @ B.astype(np.float64)
    np.testing.assert_allclose(C, ref, rtol=1e-4, atol=1e-3)
    return f"{2 * M * N * K / dt / 1e9:.0f} GFLOP/s (naive 16x16 tiles)"


@case("shared-memory reduction")
def _():
    n = 1 << 22
    x = rng.random(n, dtype=np.float32)
    dx, dout = ctx.to_device(x), ctx.empty(1, np.float32).zero()
    blocks = (n + 511) // 512
    par.function("reduce_smem")((blocks,), (256,), dx, dout, np.int32(n), shared_mem=256 * 4)
    got, ref = float(dout.to_host()[0]), float(x.astype(np.float64).sum())
    assert abs(got - ref) / ref < 1e-5, f"{got} vs {ref}"


@case("histogram with shared atomics vs numpy.bincount")
def _():
    n = 5_000_000
    data = rng.integers(0, 256, n, dtype=np.uint8)
    dd, db = ctx.to_device(data), ctx.empty(256, np.uint32).zero()
    par.function("histogram")((240,), (256,), dd, db, np.int32(n))
    np.testing.assert_array_equal(db.to_host(), np.bincount(data, minlength=256).astype(np.uint32))


@case("integer ops vs NumPy (popcount, shifts, mulhi)")
def _():
    n = 4096
    x = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
    dx, do = ctx.to_device(x), ctx.empty(n * 16, np.uint32)
    basic.function("int_ops")((n // 128,), (128,), dx, do, np.int32(n))
    o = do.to_host().reshape(n, 16)
    i = np.arange(n, dtype=np.uint32)
    popc = np.array([bin(int(v)).count("1") for v in x], dtype=np.uint32)
    np.testing.assert_array_equal(o[:, 0], popc)
    np.testing.assert_array_equal(o[:, 5], ((x.astype(np.uint64) * 0x9E3779B9) >> 32).astype(np.uint32))
    np.testing.assert_array_equal(o[:, 9], x >> (i & 31))
    np.testing.assert_array_equal(o[:, 10], (x.astype(np.uint64) << (i & 31)).astype(np.uint32))


print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
