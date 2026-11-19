# Running CUDA Programs on Apple Silicon by JIT-Translating PTX to Metal

*Translating NVIDIA PTX to Metal at load time, and what it takes to make an Apple GPU answer to `cuLaunchKernel`.*

> **Draft. Scope note before the fun part.** The programs in this article are macOS programs that use the
> CUDA Driver/Runtime API and carry their GPU code as PTX: C/C++ executables, CUDA C++ applications, and
> Python scripts that `dlopen("libcuda.so.1")`. Their source and their GPU code stay untouched; the PTX is
> JIT-translated to Metal when the program loads it. What this does **not** do is run a Linux ELF or a
> Linux PyPI wheel (`torch`, `cupy`…) on a Mac: macOS can't load those files, no matter which GPU library
> sits underneath. I'll come back to PyTorch at the end.

---

## The idea

Every CUDA program talks to the GPU through two narrow C interfaces: the **Driver API**
(`cuInit`, `cuMemAlloc`, `cuModuleLoadData`, `cuLaunchKernel`, …) and the **Runtime API** layered on top
(`cudaMalloc`, `<<<grid, block>>>`). The GPU code itself usually ships as **PTX**, NVIDIA's virtual ISA,
in addition to (or instead of) SASS machine code. The NVIDIA driver JIT-compiles that PTX for whatever
GPU it finds.

So if you implement those two interfaces and bring your own PTX JIT, the program can't tell the
difference. That's the whole trick, and it's what `libmetalcuda.dylib` does:

```
program ──cu*/cuda*──▶ libmetalcuda.dylib ──PTX→MSL──▶ Metal ──▶ M3 Max GPU
```

Three pieces:

1. **An interposer** that makes the process find this library whenever it asks for CUDA.
2. **A CUDA runtime on Metal:** contexts, memory, streams, events, modules, launches.
3. **A PTX → Metal Shading Language compiler** that runs inside `cuModuleLoadData`.

Total: about 5,500 lines of C/Objective-C, plus 2,000 lines of tests.

## Part 1: getting loaded

There are three ways a program can reach "CUDA", and each needs its own answer.

**Linked programs.** A binary built against CUDA records `@rpath/libcudart.12.dylib` or
`libcuda.1.dylib` as a dependency. The build produces those names as tiny re-export stubs of
`libmetalcuda.dylib`, so dyld is satisfied with no interposition needed.

**`dlopen` with hard-coded Linux names.** Python's ctypes, plugin loaders and a lot of glue code do
`dlopen("libcuda.so.1")`. On macOS that just fails. With `DYLD_INSERT_LIBRARIES`, the library is
loaded before the program, and its `__DATA,__interpose` section tells dyld to swap `dlopen` everywhere in
the process:

```c
#define DYLD_INTERPOSE(_replacement, _replacee)                         \
    __attribute__((used)) static struct { const void *r, *e; }         \
    _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = \
        {(const void *)&_replacement, (const void *)&_replacee};

static void *mc_dlopen(const char *path, int mode) {
    if (path && mc_is_cuda_library(path))      /* libcuda.so*, libcudart*, nvcuda.dll, CUDA.framework */
        return dlopen(self_path(), mode);      /* -> libmetalcuda.dylib */
    return dlopen(path, mode);
}
DYLD_INTERPOSE(mc_dlopen, dlopen)
```

Interposition never applies to the interposing image itself, so the `dlopen` inside `mc_dlopen` is the
real one. The test suite runs an unmodified Linux-style script that hard-codes `libcuda.so.1`. It fails
with `OSError` under plain Python and computes the right SAXPY under `python -m metalcuda`, which just
re-execs the interpreter with the right environment.

One trap: **System Integrity Protection strips `DYLD_*` variables** for system binaries. `/usr/bin/python3`
will silently ignore the injection, but a Homebrew or venv interpreter works.

**The compiler's private ABI.** A `.cu` file compiled by nvcc (or clang) never calls `cuModuleLoadData`.
Instead, a static constructor calls `__cudaRegisterFatBinary` with a pointer to the embedded fat binary,
then `__cudaRegisterFunction` for every kernel, mapping the host-side stub's address to the device name.
A launch `k<<<g, b, smem, s>>>(args)` turns into `__cudaPushCallConfiguration(g, b, smem, s)`, and the
stub then calls `__cudaPopCallConfiguration` and `cudaLaunchKernel(stub, …, args, …)` (older compilers
use `cudaConfigureCall`/`cudaSetupArgument`/`cudaLaunch`). Implement these, and parse NVIDIA's fat
binary container (magic `0xBA55ED50`, PTX entries optionally LZ4-compressed), and the kernels are found
lazily on first launch, just like CUDA 12's lazy loading.

To test this without NVIDIA's toolchain (there is no CUDA for macOS anymore), I compiled real CUDA C++ with
**clang's NVPTX backend**, which ships with Homebrew LLVM. A device pass makes PTX, a small script wraps
it in a fat binary, and a host pass embeds it with `-fcuda-include-gpubinary`. That is clang's own CUDA
pipeline with `ptxas` swapped out. The resulting Mach-O imports exactly what an nvcc binary imports.

## Part 2: memory, or the trick that made everything else easy

The first design question is what a `CUdeviceptr` *is*. CUDA programs do pointer arithmetic on device
pointers, store them inside device memory (arrays of pointers, linked lists, structs), and pass them as
kernel arguments. If device pointers were handles that needed translation, every one of those cases would
break.

Metal 3 has the answer: `MTLBuffer.gpuAddress`, a real 64-bit GPU virtual address, and MSL lets you turn
an integer into a `device T*`. So:

```objc
void *mem = mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, -1, 0);
id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:mem length:len
                    options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                    deallocator:^(void *p, NSUInteger l){ munmap(p, l); }];
*dptr = b.gpuAddress;          // this *is* the CUdeviceptr
```

and in the generated kernel:

```metal
ulong p = *(constant ulong*)(__params + 0);       // kernel argument
float x = *(device float*)(p + 4ul * i);           // just works
```

Two consequences. First, because the GPU can dereference *any* allocation, every buffer has to be resident
for every dispatch. An `MTLResidencySet` attached to the command queue does that with no per-dispatch cost.
Second, on Apple Silicon the CPU sees the same pages, so `cuMemcpyHtoD` is literally `memcpy` into the
mapping (about 55 GB/s), and `cudaMallocManaged` memory needs no page migration. Managed and pinned host
allocations return CPU addresses, and launch-time argument rewriting maps them to GPU addresses.

## Part 3: PTX to Metal

PTX looks friendly, typed registers and three-address code, but three things make translating it
interesting.

### Types are a suggestion

PTX registers have declared types, but instructions may use any register of the right *size*. Here is
what clang actually emits for `c[i] = a[i] + b[i]`:

```ptx
ld.global.b32   %r6, [%rd3];
ld.global.b32   %r7, [%rd2];
add.f32         %r8, %r6, %r7;      // float add on .b32 registers
st.global.b32   [%rd1], %r8;
```

The translator keeps every register in its declared storage type and has every instruction read its operands
*as its own type*: same-size int↔float is `as_type<>` (a bit-cast), different-size integers are value
casts (which gets sign extension right for free), and results are cast back on write:

```metal
v_r8 = as_type<uint>((as_type<float>(v_r6) + as_type<float>(v_r7)));
```

The Metal compiler erases all of it. The same rule covers f16 in 16-bit registers, `0f3F800000` float
literals in integer moves, and 64-bit packs/unpacks. `f64` is the one real gap: Metal has no double. I
compute in fp32 but keep binary64 in memory (converting on load/store), so data structures stay
compatible and the precision loss is announced, not silent.

### MSL has no `goto`

PTX control flow is labels and predicated branches. MSL has no `goto` and doesn't even allow labeled
statements (I checked: `error: 'goto' is not supported in Metal`). My first version did what everyone does
first: a `for(;;) switch (block)` dispatch loop. It was correct on every test and **4–5× slower** than
hand-written Metal on a shared-memory reduction: 40 GB/s against 174 in the same session. The GPU compiler can't see through
a switch-loop, so all the structure CUDA's compiler worked to produce gets thrown away.

For kernels with barriers or warp intrinsics, the dispatch loop also has a correctness problem. SIMT
hardware reconverges divergent threads at structured join points. A dispatch loop has no such points, so
I had to schedule it "min-PC" style: each SIMD-group executes the lowest-numbered pending block, numbered
in a loop-aware reverse post-order. That works, but it's a workaround.

The real fix is **structurizing**: rebuild `for`/`if` from the CFG. I implemented Norman Ramsey's
dominator-tree algorithm from *Beyond Relooper* (ICFP 2022). Loop headers become `for(;;)`, back edges
become `continue`, nodes with several forward predecessors become the follower of a
`do { … } while (false)` block (so forward branches to them are `break`), and everything else is inlined
at its single predecessor. Two adaptations were needed:

* **C has no labeled `break`.** A jump that leaves several constructs at once sets a variable and breaks,
  and each construct it crosses forwards it outward:
  `if (__jmp) { if (__jmp == 3u) { __jmp = 0u; break; } else break; }`
* **Don't inline a loop's exit into the loop.** For sequential code it's fine to emit a loop's exit block
  inside the loop body (it ends in a branch anyway). For a GPU it means the code after a loop, typically a
  `__syncthreads()`, runs per thread at different iterations. Exits are therefore placed after the
  outermost loop they leave, which is where CUDA programmers expect convergence. Honest footnote: in my
  test the M3's barrier tolerated the naive placement too. I keep the relocation because it matches the
  CUDA model and costs nothing.

Irreducible CFGs (rare, and never produced from structured source in my tests) still get the dispatch loop,
and there's a hand-written irreducible PTX kernel in the test suite to keep that path honest.

Result: the reduction went from 40 to 130–217 GB/s (the same algorithm hand-written in MSL does 175–262
GB/s on this machine), and a naive tiled SGEMM from 0.5 to 1.2–1.57 TFLOP/s. Correctness didn't move.

### The 976 pixels

My favorite bug wasn't a bug in the translator. The runtime test renders a 1024×768 Mandelbrot set on
the GPU and compares it with the CPU: **976 pixels were different**. Mandelbrot is chaotic at the boundary,
so one ulp anywhere becomes a different iteration count. "Floating point, close enough" was tempting.

But PTX is precise about floating point. `fma.rn.f32` is a fused multiply-add, and a plain `mul.f32` followed
by `add.f32` is not one. Clang had already fused what it wanted to. Then the Metal compiler, by default,
fused some of the *remaining* mul/add pairs again. So I added `#pragma METAL fp contract(off)` to every
generated library, and the mismatch… stayed at 976.

Flipping the pragma to `fast` gave **zero** mismatches. That inversion was the clue: the *reference* was
wrong. The CPU code in the test was compiled by clang's CUDA host pass, which also defaults to
`-ffp-contract=fast`, so the "reference" had been FMA-contracted by the host compiler differently from the
PTX. Pinning the reference with `#pragma clang fp contract(off)` and writing it as the PTX's literal
operation sequence gives 0 / 786,432 mismatches with contraction off on the GPU. The GPU now reproduces the
PTX's operation sequence **bit for bit**, which is a much stronger statement than "within tolerance".
(`METALCUDA_FP_CONTRACT=fast` restores the ptxas-like behaviour.)

### Everything else

The long tail is what makes real compiler output run:

* **Shared memory** becomes one `threadgroup` buffer: static variables first, then `extern __shared__`.
* **Local memory** (`.local` stack frames) becomes a per-thread array.
* **Generic addresses** that may point into shared memory (clang `-O0` produces lots of these via
  `cvta.shared`) get a tag in the top 16 bits, and loads, stores and atomics check it at runtime.
* `shfl.sync` is implemented with its exact clamp/segment-mask semantics on top of `simd_shuffle`. Apple
  SIMD-groups are 32 wide, like warps.
* Float atomics in threadgroup memory don't exist in MSL, so they become CAS loops. 64-bit atomic add
  becomes two 32-bit atomics with carry (exact sum).
* `__syncthreads_count` is `bar.red.popc`, which needs a scratch word and three barriers.
* PTX's `mul24` is defined for any input, while MSL's is undefined outside ±2²³. That one was caught by a
  random-input test, not by reading the spec.

The test suite compiles the same kernels at `-O3`, `-O1` and `-O0`, because the PTX differs wildly
(the `-O0` build of one test file has 471 generic loads/stores and 29 `cvta`s). All 25 tests pass on all three.

## Part 4: does it work, and is it fast?

`make test` from a clean tree runs:

* the Driver API walk-through with nvcc-format PTX;
* a 25-test conformance suite × 3 optimization levels, each against a CPU reference;
* a normal CUDA C++ application with `<<<>>>` launches, templates, `__constant__` and `__device__`
  symbols, managed memory, streams, events and error handling, built with both launch ABIs;
* Python: the Linux-style `libcuda.so.1` script, `metalcuda.enable()`, and NumPy-verified
  vecadd/SAXPY/SGEMM/reduction/histogram.

Numbers on an M3 Max (30-core GPU):

| | MetaCuda | native Metal |
|---|---|---|
| `cuLaunchKernel` host cost | 0.2–0.5 µs | — |
| dependent tiny launches | 2.4–3.3 µs | 1.25 µs |
| launch + sync round trip | 166–193 µs | 154–156 µs |
| tree reduction, 16 M floats | 130–217 GB/s | 175–262 GB/s |
| naive tiled SGEMM 1024³ | 1.2–1.57 TFLOP/s | |
| first / cached module load | ~400 ms / ~2–3 ms | |

Launches are batched into one serial compute encoder with buffer barriers between dispatches. Cold JIT
cost (Metal compiling the MSL) is paid once: the MSL and a Metal binary archive land in
`~/.cache/metalcuda`.

A cautionary tale about benchmarking: halfway through, every launch suddenly cost 90 µs. I tried batch sizes, event chaining, barriers and
allocation strategies before writing a 30-line pure-Metal benchmark. It
was just as slow. The machine was in a degraded GPU state. Later in the session native Metal was back to
1.3 µs, and so was MetaCuda. **Always measure the baseline in the same session.**

## What it doesn't do (and what PyTorch would need)

* **Foreign binaries.** Linux `.so` files and wheels can't load on macOS. That's an OS boundary, not a
  GPU one.
* **SASS-only cubins.** If a binary ships only NVIDIA machine code and no PTX, there's nothing portable to
  translate. nvcc's default `-arch=sm_XX` does embed PTX for that architecture, and the loader says
  clearly when PTX is missing.
* **Tensor cores, textures, non-inlined device calls, dynamic parallelism, grid sync** aren't translated yet.
  `mma.sync` → `simdgroup_matrix` is the obvious next step.
* **Hardware differences** are reported honestly: 32 KiB of threadgroup memory (kernels asking for more
  fail at launch with the CUDA error you'd expect), fp32 instead of fp64, limited 64-bit atomics, and
  streams sharing one queue.

PyTorch is the elephant in the room. `torch` for macOS is built *without* its CUDA backend, so there's no
`cudaLaunchKernel` call anywhere to intercept. Getting `torch.cuda` would mean compiling PyTorch's CUDA
backend for macOS against these headers (feasible in principle: it's mostly PTX-emitting CUDA C++), *and*
providing cuBLAS/cuDNN, which are closed libraries that ship as SASS. That's a project, not a weekend. But the
foundation (the runtime, the ABI, the memory model and a PTX JIT that is bit-exact on what it supports)
is now there and tested.

## Try it

```sh
make test     # everything, from scratch
make bench    # numbers + native Metal baseline on your machine
build/bin/mc-ptx2msl -c kernel.ptx   # see the Metal your PTX becomes
```

---

*Notes for the final version: add the generated-MSL listing of the reduction kernel before/after
structurizing, a diagram of the dominator tree for it, and re-run `make bench` on release day.*
