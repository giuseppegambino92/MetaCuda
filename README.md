# MetaCuda — `libmetalcuda.dylib`

**Run CUDA programs on Apple Silicon GPUs.** MetaCuda is a drop-in implementation of the CUDA
Driver and Runtime APIs for macOS. It JIT-translates the PTX embedded in a CUDA program into
Metal Shading Language and runs it on the Apple GPU through Metal, with device memory living
in unified memory.

```
$ ./build/bin/test_cuda_basic
driver 12.4 | 1 device(s) | Apple M3 Max (MetaCuda) | 28.1 GiB | sm_80 | 30 SMs
vectorAdd: 1048576 elements, 4096 blocks x 256 threads, device ptrs 0x10000080000 0x10000488000 0x10000890000 -> PASSED
```

What runs **unmodified** today (all verified by `make test`):

| Program | How it reaches MetaCuda |
|---|---|
| A C/C++ program using the **Driver API** (`cuModuleLoadData`, `cuLaunchKernel`, …) | links `libcuda.dylib` (provided) |
| A **CUDA C++ program** (`<<<>>>`, templates, `__constant__`, `__device__`, managed memory, streams, events) compiled for macOS from `.cu` | links `libcudart.dylib` (provided). Its embedded fat binary is registered through `__cudaRegisterFatBinary` and launched through `__cudaPushCallConfiguration`/`cudaLaunchKernel`, the same ABI nvcc emits. |
| A **Python** script that does `ctypes.CDLL("libcuda.so.1")` like on Linux | `python -m metalcuda script.py` injects the library; a dyld interposer redirects the `dlopen` |
| Python code that wants an explicit opt-in | `import metalcuda; metalcuda.enable()` |

What it does **not** do (see [Limitations](#limitations)): run Linux/Windows executables or
wheels (wrong OS binary format), or run kernels that only ship as NVIDIA SASS machine code
(PTX is required). In particular there is no CUDA-enabled PyTorch for macOS for it to plug into.

---

## Quick start

Requirements: Apple Silicon Mac, Xcode command line tools. Tests also use Homebrew LLVM
(`brew install llvm`, for clang's NVPTX backend) and `uv` (for a NumPy venv).
**No NVIDIA software is needed.**

```sh
make            # build/lib/libmetalcuda.dylib + libcuda/libcudart names + tools
make test       # C driver tests, CUDA runtime app, Python tests  → "ALL TESTS PASSED"
make bench      # micro-benchmarks + native Metal baseline
```

### Use it

```sh
# 1. Programs linked against CUDA
clang my_driver_app.c -Iinclude -Lbuild/lib -lcuda -Wl,-rpath,$PWD/build/lib

# 2. Inject into an existing process (non-SIP binaries only)
DYLD_INSERT_LIBRARIES=$PWD/build/lib/libmetalcuda.dylib ./some_program

# 3. Python
PYTHONPATH=python python -m metalcuda my_cuda_script.py     # injected (unmodified scripts)
python -c "import metalcuda; metalcuda.enable(); import ctypes; ctypes.CDLL('libcuda.so.1')"
```

### Compile a `.cu` file for macOS without the CUDA toolkit

`include/` provides `cuda_runtime.h` (with the device-side language support), so clang can compile
ordinary CUDA C++. This is exactly what clang's CUDA driver does internally, done in three steps
because NVIDIA's `ptxas`/`fatbinary` aren't available on macOS:

```sh
CLANG=$(brew --prefix llvm)/bin/clang++
FLAGS="-x cuda -nocudainc -nocudalib --cuda-gpu-arch=sm_80 -Iinclude"
$CLANG $FLAGS --cuda-device-only -Xclang -target-feature -Xclang +ptx75 -O3 -S app.cu -o app.ptx   # device → PTX
python3 tools/mkfatbin.py --compress app.ptx -o app.fatbin                                         # PTX → fat binary
$CLANG $FLAGS --cuda-host-only -Xclang -target-sdk-version=12.4 -O2 \
       -Xclang -fcuda-include-gpubinary -Xclang app.fatbin -c app.cu -o app.o                     # host, embeds fatbin
$CLANG app.o -Lbuild/lib -lcudart -Wl,-rpath,$PWD/build/lib -o app
```

The resulting executable is a normal CUDA runtime application: it imports
`__cudaRegisterFatBinary`, `__cudaPushCallConfiguration`, `cudaLaunchKernel`, `cudaMalloc`, …
from `@rpath/libcudart.12.dylib`.

---

## Architecture

```
 CUDA program (C/C++/Python)                         Apple GPU
        │  cuLaunchKernel / <<<>>> / ctypes                ▲
        ▼                                                   │
 ┌──────────────── libmetalcuda.dylib ─────────────────┐    │
 │ interpose.c   dyld __interpose: dlopen("libcuda.so*",│    │
 │               "libcudart*", "nvcuda.dll") → self      │    │
 │ cudart.c      Runtime API + nvcc/clang launch ABI     │    │
 │ runtime.m     Driver API on Metal ────────────────────┼────┘
 │   memory  : MTLBuffer(bytesNoCopy, shared) over mmap; │
 │             CUdeviceptr == MTLBuffer.gpuAddress        │
 │   residency: MTLResidencySet on the queue              │
 │   launch  : batched serial compute encoder + barriers  │
 │   modules : fatbin.c → PTX → ptx_to_metal.c → MSL      │
 │             → MTLLibrary → MTLComputePipelineState     │
 │             cache: ~/.cache/metalcuda (.metal + binary archive)
 └───────────────────────────────────────────────────────┘
```

Key design decisions:

* **Device pointers are GPU virtual addresses.** `cuMemAlloc` returns `MTLBuffer.gpuAddress`, and the
  translated kernels cast 64-bit integers straight to `device T*` (MSL 3.x). Pointer arithmetic, pointers
  stored in memory (arrays of pointers, linked structures) and `cuMemGetAddressRange` behave like CUDA.
  The same physical pages are CPU-mapped, so `cuMemcpy*` is a plain `memcpy` (52–58 GB/s measured) and
  managed memory needs no migration at all.
* **Residency.** Kernels can dereference *any* allocation, so every allocation is added to an
  `MTLResidencySet` attached to the command queue (with a `useResource:` fallback for pre-macOS-15).
* **One kernel ABI.** `buffer(0)` holds the CUDA parameter block in CUDA's own layout (so
  `CU_LAUNCH_PARAM_BUFFER_POINTER` and the legacy `cudaSetupArgument` path work), followed by a hidden
  `{globals base, dynamic smem size}` block. `threadgroup(0)` holds static + dynamic shared memory.
* **PTX → MSL translation** (`src/ptx_to_metal.c`):
  * registers keep their PTX storage class. Every instruction reads its operands *as its own type*
    (bit-casts between same-size int/float, value casts between ints), which is PTX's "bit-size types
    are compatible with any type of the same size" rule.
  * **Structured control flow.** MSL has no `goto` and no labels. Reducible CFGs are rebuilt as
    `for`/`do-while`/`if` with a dominator-tree algorithm (Ramsey, *Beyond Relooper*, ICFP 2022),
    extended with emulated multi-level `break`/`continue` and loop exits placed *after* the loop (SIMT
    reconvergence). Irreducible CFGs fall back to a convergence-aware ("min-PC") block dispatch loop.
    Structurizing made the shared-memory reduction 3–5× faster and SGEMM 2.5–3× faster than the dispatch loop.
  * generic addresses get a tag in the top 16 bits when a kernel converts shared/local/param addresses
    to generic ones (`cvta`), and loads, stores and atomics dispatch on the tag. That is what makes clang `-O0`
    output work.
  * **Bit-exact floating point.** MSL is compiled with `#pragma METAL fp contract(off)` and safe math,
    so only the PTX's explicit `fma`/`mad` are fused. A 1024×768 Mandelbrot (up to 256 chaotic iterations
    per pixel) matches a CPU reference that follows the PTX operation sequence, **0 of 786,432 pixels
    different**. `METALCUDA_FP_CONTRACT=fast` lets Metal contract like ptxas may.
  * **IEEE binary64 in software.** Metal has no `double`. f64 values stay binary64 bit patterns in `ulong`
    registers and every operation goes through a soft-float library in MSL (`src/softfloat64.metal`:
    SoftFloat-style add/sub/mul/fma with an exact 106-bit product, bit-serial division and square root,
    round-to-nearest-even, subnormals, signed zeros, NaN/inf). It is checked bit for bit against the CPU on
    17 operations × 262,144 inputs, including NaN, infinities, subnormals and catastrophic cancellation. The
    cost is real: about 4.6× slower than fp32 on a memory-bound kernel, and more on compute-bound ones.
    `METALCUDA_F64=fast` computes f64 in fp32 instead (layout stays binary64).
  * **64-bit atomics** (Metal has no 64-bit CAS): every 64-bit atomic, `double` `atomicAdd` included, is a
    read-modify-write under a spinlock from a hashed lock table. One lane per SIMD-group is elected with a
    ballot to take a lock, so no lane ever spins on a lock held by its own SIMD-group. That keeps it
    deadlock-free, which the first, simpler retry loop was not.
* **Caching.** Module key = SHA-256(translator version, fp policy, PTX). The MSL is kept for inspection and
  the compiled pipelines go into a Metal binary archive: module load ~400 ms cold → ~2–3 ms warm.

### Supported surface

* **Driver API (132 entry points):** init/device/attributes, primary + regular contexts, streams, events
  (with GPU timing), `cuMemAlloc/Free/AllocManaged/HostAlloc/HostRegister/GetInfo/GetAddressRange`,
  all `cuMemcpy*`/`cuMemset*` (sync + async), virtual memory management
  (`cuMemAddressReserve/Create/Map/SetAccess/Unmap`), `cuPointerGetAttribute`, modules
  (`cuModuleLoad/LoadData/LoadDataEx/LoadFatBinary/GetFunction/GetGlobal`), `cuLaunchKernel` (array and
  `extra` parameter protocols), function attributes, occupancy, `cuGetProcAddress`.
* **Runtime API (79 entry points):** device/properties, memory (incl. managed/pinned/symbols), streams,
  events, `cudaLaunchKernel`, both launch ABIs (`__cudaPushCallConfiguration` and
  `cudaConfigureCall`/`cudaSetupArgument`/`cudaLaunch`), fatbin/function/variable/managed-variable
  registration, error handling.
* **PTX:** integer/float/f16/f16x2/bf16 arithmetic, all comparison modes (ordered/unordered), `cvt`
  with rounding and saturation, `ld/st` in all state spaces incl. vectors and generic addressing,
  `cvta`, atomics/reductions (add/min/max/inc/dec/exch/cas/and/or/xor; 32-bit, f32, and all 64-bit ops incl. f64 add,
  on device and shared memory), f64 arithmetic/conversions/comparisons (exact IEEE),
  `bar.sync`, `bar.red` (`__syncthreads_count/and/or`), `membar`/`fence`, `shfl.sync` (exact clamp/segment
  semantics), `vote`, `redux`, `activemask`, `prmt`, `lop3`, `shf`, `bfe`/`bfi`, `bfind`, `popc`, `clz`,
  `brev`, `dp4a`, `mul24`, `cp.async` (executed synchronously), `.local` stack frames, module `.global`/`.const`
  variables with initializers, `.shared` static + `extern`.
* **Images:** raw PTX, fat binary containers (incl. LZ4-compressed entries), fatbin wrappers.

---

## Verification

`make test` (from a clean tree, cold kernel cache):

| Test | What it covers |
|---|---|
| `tests/test_cuda_basic.c` | Phase-4 Driver API walk-through with nvcc-format PTX (vectorAdd, 1 M elements) |
| `tests/test_driver_suite.c` | 31 tests × 3 builds of the same kernels (clang `-O3`, `-O1`, `-O0`): integer/bit/float semantics, conversions, structs by value, f64, 3-D grids, shared-memory reduction, warp shuffles/votes, histogram (shared + global atomics), scan, tiled SGEMM, atomics (CAS loops, inc wrap, 64-bit), local memory, `bar.red`, globals/constants, managed/pinned/registered memory, VMM, events, error paths, `cuGetProcAddress`, 10 k dependent launches, divergent-loop + barrier, an irreducible CFG, async-copy ordering and host-buffer reuse, reported capabilities; 64-bit atomics (returned old values must be an exact permutation, exchange must conserve every value, CAS loops, min/max, shared memory, `double` `atomicAdd`); IEEE binary64 bit-exactness |
| `tests/test_runtime_app.cu` | an ordinary CUDA C++ app, built with both launch ABIs: bit-exact Mandelbrot, template kernels on a stream, softmax with dynamic shared memory, `cudaMemcpyToSymbol`/`FromSymbol`, managed memory, error idioms |
| `tests/test_python.py` | a Linux-style `ctypes.CDLL("libcuda.so.1")` script fails on plain macOS and passes under `python -m metalcuda`; `metalcuda.enable()`; NumPy-checked vecadd, SAXPY, SGEMM, reduction, histogram, integer ops |

Every GPU result is compared against a CPU reference (exact for integer/bitwise work and wherever the
operation sequence is deterministic, tight tolerances for approximate math intrinsics).

### Benchmarks (M3 Max, 30-core GPU; `make bench`)

| | MetaCuda | native Metal |
|---|---|---|
| `cuLaunchKernel` host cost | 0.2–0.5 µs | — |
| dependent tiny launches, batched | 2.4–3.3 µs | 1.25 µs |
| launch + synchronize round trip | 166–193 µs | 154–156 µs |
| `cuMemcpyHtoD` / `DtoH`, 256 MiB | 52–58 GB/s | (CPU memcpy) |
| vecadd, 16 M floats | 245–255 GB/s effective | |
| shared-memory tree reduction, 16 M floats | 130–217 GB/s | 175–262 GB/s (same algorithm hand-written in MSL) |
| naive 16×16-tiled SGEMM, 1024³ | 1.2–1.57 TFLOP/s | |
| Mandelbrot 1024×768×256 | 15–30 G iterations/s | |
| `f64_ops` (mul+fma+sqrt+sub), 4 M doubles | 1.55 G elements/s exact IEEE; 7.1 G with `METALCUDA_F64=fast` (fp32) | |

Numbers vary between runs with GPU clock state. During one session the whole machine (native Metal
included) was 10–50× slower per dispatch for a while, so measure on your own system before quoting.

---

## Limitations

* **Executables from other operating systems.** A Linux ELF or Windows PE program (or a PyPI wheel with
  Linux `.so` files: `torch`, `cupy`, `bitsandbytes` CUDA builds) cannot be loaded by macOS at all. MetaCuda
  runs macOS programs that use the CUDA APIs. PyTorch for macOS is built without its CUDA backend, so there
  is nothing to intercept. Making `torch.cuda` work would mean building PyTorch's CUDA backend for macOS
  against these headers, plus implementing cuBLAS/cuDNN.
* **SASS-only binaries.** Cubins contain NVIDIA machine code. MetaCuda needs PTX (nvcc embeds it with
  `-gencode arch=compute_XX,code=compute_XX`, the default for the newest arch). The loader reports this
  explicitly. ZSTD-compressed fatbin entries (CUDA ≥ 12.8 option) are not supported yet.
* **Not translated:** tensor-core instructions (`mma`, `wmma`, `ldmatrix`), textures/surfaces, calls to
  non-inlined device functions (`call` other than `vprintf`, which is dropped), indirect branches,
  split barriers, cluster/TMA features, dynamic parallelism, grid-wide cooperative sync.
* **f64 is exact but slow** (soft-float, see above). Rounding-mode modifiers other than round-to-nearest on
  f64 arithmetic (`add.rz.f64`, …) are treated as round-to-nearest.
* **64-bit atomics are correct but serialized per address.** Each attempt costs ~0.2 µs, so a 64-bit CAS retry
  loop with many threads on one address degrades quadratically (one address, 4,096 threads: 1.7 s;
  16,384 threads: 31 s). They are not atomic with respect to 32-bit atomics on the halves of the same word.
* **Hardware:** threadgroup memory is 32 KiB (CUDA kernels requesting more fail with
  `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`). Named barriers become full barriers.
* **Scheduling:** all streams share one in-order Metal queue. This is a legal CUDA schedule, but kernels on
  different streams never overlap: two independent kernels on two streams take about as long as running
  them back to back. The device therefore reports `concurrentKernels = 0`, `asyncEngineCount = 0` and
  `deviceOverlap = 0`. Async memcpys complete synchronously and in stream order, so the host buffer can be
  reused as soon as the call returns. `cuMemFree` synchronises.
* **No isolation between kernels in a process.** Device pointers are raw GPU addresses and every allocation
  is resident, so a kernel can read or write any allocation in its process, not just the ones passed as
  arguments. Out-of-range accesses are not reported: in a quick probe, writes to unmapped addresses returned
  `CUDA_SUCCESS` with no fault. CUDA doesn't isolate allocations within a context either; isolation from
  other processes relies on macOS giving each process its own GPU address space (not tested here). Don't run untrusted kernels in a
  process that holds data they shouldn't see.
* **Interposition** needs a non-SIP process: `/usr/bin/python3` and other system binaries ignore
  `DYLD_INSERT_LIBRARIES`. Use Homebrew/pyenv/venv Python.

---

## Environment variables

| Variable | Effect |
|---|---|
| `METALCUDA_LOG=1` / `2` / `3` | load/compile/redirect info / per-allocation & launch / per-command-buffer GPU time |
| `METALCUDA_CACHE_DIR` | kernel cache location (default `~/.cache/metalcuda`, or `$XDG_CACHE_HOME/metalcuda`) |
| `METALCUDA_SYNC=1` | synchronise after every launch (debugging) |
| `METALCUDA_FP_CONTRACT=fast` | allow Metal to contract mul+add into FMA (default: off, bit-exact with the PTX) |
| `METALCUDA_F64=fast` | compute f64 in fp32 (default: exact IEEE binary64 in software) |
| `METALCUDA_NO_STRUCTURIZE=1` | force the block-dispatch loop (debugging the structurizer) |
| `METALCUDA_DISABLE_REDIRECT=1` | keep the library injected but stop redirecting `dlopen` |
| `METALCUDA_LIB` | Python: path to `libmetalcuda.dylib` |

Inspect what the translator produces: `build/bin/mc-ptx2msl -c kernel.ptx` (prints the MSL and compiles it
with Metal), or look at `~/.cache/metalcuda/<hash>.metal`.

## Layout

```
include/        cuda.h, cuda_runtime.h, cuda_runtime_api.h, cudart.h, vector_types.h, metalcuda_device.h
src/            ptx_to_metal.c (translator + structurizer), softfloat64.metal (IEEE binary64 in MSL),
                runtime.m (Driver API on Metal),
                cudart.c (Runtime API + launch ABI), fatbin.c (fatbin/LZ4), interpose.c (dyld interposer)
python/metalcuda/   enable(), `python -m metalcuda`, driver.py (NumPy-friendly Driver API wrapper)
tools/          mc-ptx2msl, mkfatbin.py (fat binary packer with LZ4), metal_baseline.m, embed_msl.py
tests/          test_cuda_basic.c, test_driver_suite.c, test_runtime_app.cu, test_python.py, bench.c, kernels/*.cu
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
