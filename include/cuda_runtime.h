/*
 * MetaCuda — cuda_runtime.h
 *
 * Host code: the CUDA Runtime API (implemented by libmetalcuda / libcudart.dylib).
 * CUDA mode (clang -x cuda): additionally provides the device-side language
 * support normally supplied by NVIDIA's headers, so ordinary .cu files compile
 * with clang's NVPTX backend without the CUDA toolkit.
 */
#ifndef METACUDA_CUDA_RUNTIME_H
#define METACUDA_CUDA_RUNTIME_H

#if defined(__CUDA__)
#include "metalcuda_device.h"
#endif

#include "cuda_runtime_api.h"

#endif
