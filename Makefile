# MetaCuda — build libmetalcuda.dylib, tools and tests.
#
#   make            build the library, CUDA-name shims and tools
#   make test       build + run the full verification suite (C, CUDA runtime app, Python)
#   make clean
#
# Requirements: Xcode command line tools (Apple clang + Metal.framework).
# Tests additionally use Homebrew LLVM (clang with the NVPTX backend) to compile
# the .cu test programs to genuine PTX — no NVIDIA toolkit is needed.

BUILD    := build
OBJ      := $(BUILD)/obj
LIB      := $(BUILD)/lib
BIN      := $(BUILD)/bin
PTXDIR   := $(BUILD)/ptx

CC       := clang
CFLAGS   := -O2 -g -Wall -Wextra -Wno-unused-parameter -fvisibility=hidden -Iinclude -Isrc -I$(OBJ)
OBJCFLAGS:= $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Metal -framework Foundation -framework QuartzCore -framework IOKit

LLVM_PREFIX ?= $(shell brew --prefix llvm 2>/dev/null || echo /opt/homebrew/opt/llvm)
CUDACXX  ?= $(LLVM_PREFIX)/bin/clang++
CUDA_ARCH ?= sm_80
CUDA_DEVICE_FLAGS := -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=$(CUDA_ARCH) -Xclang -target-feature -Xclang +ptx75 -O3 -S -Iinclude
# -target-sdk-version selects the CUDA >= 9.2 launch ABI (__cudaPushCallConfiguration + cudaLaunchKernel),
# which is what nvcc-built binaries use; the *_legacy test variant exercises the old cudaConfigureCall ABI.
CUDA_HOST_FLAGS   := -x cuda --cuda-host-only -nocudainc -nocudalib --cuda-gpu-arch=$(CUDA_ARCH) -O2 -Iinclude -Xclang -target-sdk-version=12.4
CUDA_HOST_FLAGS_LEGACY := -x cuda --cuda-host-only -nocudainc -nocudalib --cuda-gpu-arch=$(CUDA_ARCH) -O2 -Iinclude
PYTHON   ?= $(BUILD)/venv/bin/python

SRCS_C   := src/ptx_to_metal.c src/fatbin.c src/cudart.c src/interpose.c
SRCS_M   := src/runtime.m
OBJS     := $(SRCS_C:src/%.c=$(OBJ)/%.o) $(SRCS_M:src/%.m=$(OBJ)/%.o)
HEADERS  := $(wildcard src/*.h include/*.h)

DYLIB    := $(LIB)/libmetalcuda.dylib
SHIMS    := $(LIB)/libcuda.1.dylib $(LIB)/libcudart.12.dylib

KERNELS  := $(wildcard tests/kernels/*.cu)
PTX      := $(KERNELS:tests/kernels/%.cu=$(PTXDIR)/%.ptx)

.SECONDARY:
.PHONY: all lib tools tests test test-c test-app test-python clean venv bench

all: lib tools

lib: $(DYLIB) $(SHIMS)

# the binary64 soft-float library is MSL source embedded into the translator
$(OBJ)/softfloat64.inc: src/softfloat64.metal tools/embed_msl.py | $(OBJ)
	python3 tools/embed_msl.py MC_SOFTFLOAT64_MSL $< > $@

$(OBJ)/ptx_to_metal.o: $(OBJ)/softfloat64.inc

$(OBJ)/%.o: src/%.c $(HEADERS) | $(OBJ)
	$(CC) $(CFLAGS) -std=c11 -c $< -o $@

$(OBJ)/%.o: src/%.m $(HEADERS) | $(OBJ)
	$(CC) $(OBJCFLAGS) -c $< -o $@

$(DYLIB): $(OBJS) | $(LIB)
	$(CC) -dynamiclib -install_name @rpath/libmetalcuda.dylib -current_version 1.0 -compatibility_version 1.0 \
	    -o $@ $(OBJS) $(FRAMEWORKS)
	@echo "built $@ ($$(nm -gU $@ | grep -c ' _cu') cu* + $$(nm -gU $@ | grep -c ' _cuda') cuda* exports)"

# NVIDIA-named libraries for programs linked against libcuda / libcudart.
# They re-export libmetalcuda so the program records the real CUDA install names.
$(LIB)/libcuda.1.dylib: $(DYLIB)
	$(CC) -dynamiclib -install_name @rpath/libcuda.1.dylib -Wl,-reexport_library,$(DYLIB) -o $@ -L$(LIB)
	ln -sf libcuda.1.dylib $(LIB)/libcuda.dylib
	ln -sf libmetalcuda.dylib $(LIB)/libcuda.so.1
	ln -sf libmetalcuda.dylib $(LIB)/libcuda.so
$(LIB)/libcudart.12.dylib: $(DYLIB)
	$(CC) -dynamiclib -install_name @rpath/libcudart.12.dylib -Wl,-reexport_library,$(DYLIB) -o $@ -L$(LIB)
	ln -sf libcudart.12.dylib $(LIB)/libcudart.dylib
	ln -sf libmetalcuda.dylib $(LIB)/libcudart.so.12
	ln -sf libmetalcuda.dylib $(LIB)/libcudart.so

tools: $(BIN)/mc-ptx2msl

$(BIN)/mc-ptx2msl: tools/ptx2msl.m src/ptx_to_metal.c src/ptx_to_metal.h $(OBJ)/softfloat64.inc | $(BIN)
	$(CC) -O2 -fobjc-arc -Isrc -I$(OBJ) tools/ptx2msl.m src/ptx_to_metal.c -framework Metal -framework Foundation -o $@

# ---------------------------------------------------------------- tests

# .cu kernels → PTX with clang's NVPTX backend (genuine compiler output)
$(PTXDIR)/%.ptx: tests/kernels/%.cu include/metalcuda_device.h | $(PTXDIR)
	$(CUDACXX) $(CUDA_DEVICE_FLAGS) $< -o $@

# The same kernels at -O0 / -O1: very different PTX (local stack frames, generic
# addressing via cvta, unoptimised control flow) — a robustness test for the translator.
PTX_O0 := $(KERNELS:tests/kernels/%.cu=$(BUILD)/ptx-O0/%.ptx)
PTX_O1 := $(KERNELS:tests/kernels/%.cu=$(BUILD)/ptx-O1/%.ptx)
$(BUILD)/ptx-O0/%.ptx: tests/kernels/%.cu include/metalcuda_device.h
	@mkdir -p $(dir $@)
	$(CUDACXX) $(subst -O3,-O0,$(CUDA_DEVICE_FLAGS)) $< -o $@
$(BUILD)/ptx-O1/%.ptx: tests/kernels/%.cu include/metalcuda_device.h
	@mkdir -p $(dir $@)
	$(CUDACXX) $(subst -O3,-O1,$(CUDA_DEVICE_FLAGS)) $< -o $@

# Driver API test (the Phase 4 basic test) — hand-written nvcc-style PTX inside
$(BIN)/test_cuda_basic: tests/test_cuda_basic.c lib | $(BIN)
	$(CC) -O2 -Iinclude $< -L$(LIB) -lcuda -Wl,-rpath,@loader_path/../lib -o $@

# Differential test of the binary64 soft-float against the CPU's FPU (all rounding modes)
$(BIN)/test_softfloat: tests/test_softfloat.c lib | $(BIN)
	$(CC) -O1 -ffp-model=strict -Iinclude $< -L$(LIB) -lcuda -Wl,-rpath,@loader_path/../lib -o $@

# Driver API suite over clang-generated PTX
$(BIN)/test_driver_suite: tests/test_driver_suite.c lib | $(BIN)
	$(CC) -O2 -ffp-contract=off -Iinclude $< -L$(LIB) -lcuda -Wl,-rpath,@loader_path/../lib -o $@

# A complete CUDA *runtime* application (<<<>>> launches, cudaMalloc, __constant__,
# __device__ globals, events). Compiled exactly like clang's CUDA driver does it:
# device pass → PTX → fatbin; host pass embeds the fatbin and emits the
# __cudaRegisterFatBinary / __cudaPushCallConfiguration / cudaLaunchKernel ABI.
$(BUILD)/app/%.ptx: tests/%.cu include/metalcuda_device.h | $(BUILD)/app
	$(CUDACXX) $(CUDA_DEVICE_FLAGS) $< -o $@
$(BUILD)/app/%.fatbin: $(BUILD)/app/%.ptx tools/mkfatbin.py
	python3 tools/mkfatbin.py --compress $< -o $@
$(BUILD)/app/%.o: tests/%.cu $(BUILD)/app/%.fatbin
	$(CUDACXX) $(CUDA_HOST_FLAGS) -Xclang -fcuda-include-gpubinary -Xclang $(BUILD)/app/$*.fatbin -c $< -o $@
$(BUILD)/app/%_legacy.o: tests/%.cu $(BUILD)/app/%.fatbin
	$(CUDACXX) $(CUDA_HOST_FLAGS_LEGACY) -Xclang -fcuda-include-gpubinary -Xclang $(BUILD)/app/$*.fatbin -c $< -o $@
$(BIN)/%: $(BUILD)/app/%.o lib | $(BIN)
	$(CUDACXX) $< -L$(LIB) -lcudart -Wl,-rpath,@loader_path/../lib -o $@

tests: $(PTX) $(PTX_O0) $(PTX_O1) $(BIN)/test_cuda_basic $(BIN)/test_driver_suite $(BIN)/test_softfloat $(BIN)/test_runtime_app $(BIN)/test_runtime_app_legacy

# A non-SIP interpreter is required for the DYLD_INSERT_LIBRARIES test (/usr/bin/python3 strips DYLD_*).
BASE_PYTHON ?= $(firstword $(wildcard /opt/homebrew/bin/python3 /opt/homebrew/bin/python3.1[0-9] /usr/local/bin/python3))
venv: $(BUILD)/venv/.ok
$(BUILD)/venv/.ok:
	uv venv --python $(BASE_PYTHON) $(BUILD)/venv >/dev/null
	uv pip install --python $(BUILD)/venv/bin/python numpy >/dev/null
	touch $@

test-c: tests
	$(BIN)/test_cuda_basic
	$(BIN)/test_driver_suite $(PTXDIR)
	$(BIN)/test_driver_suite $(BUILD)/ptx-O0
	$(BIN)/test_driver_suite $(BUILD)/ptx-O1
	$(BIN)/test_softfloat

test-app: tests
	$(BIN)/test_runtime_app
	$(BIN)/test_runtime_app_legacy

test-python: lib $(PTX) venv
	PYTHONPATH=python $(PYTHON) tests/test_python.py

test: test-c test-app test-python
	@echo "ALL TESTS PASSED"

$(BIN)/bench: tests/bench.c lib | $(BIN)
	$(CC) -O2 -Iinclude $< -L$(LIB) -lcuda -Wl,-rpath,@loader_path/../lib -o $@
$(BIN)/metal_baseline: tools/metal_baseline.m | $(BIN)
	$(CC) -O2 -fobjc-arc $< -framework Metal -framework Foundation -framework QuartzCore -o $@
bench: $(BIN)/bench $(BIN)/metal_baseline $(PTX)
	$(BIN)/bench $(PTXDIR)
	$(BIN)/metal_baseline

$(OBJ) $(LIB) $(BIN) $(PTXDIR) $(BUILD)/app:
	mkdir -p $@

clean:
	rm -rf $(BUILD)
