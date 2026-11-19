// Module-scope variables: __constant__ tables, initialised __device__ globals.
#include <cuda_runtime.h>

__constant__ float coeffs[4] = {0.5f, -1.25f, 2.0f, 3.75f};
__device__ int counter = 0;
__device__ unsigned lut[8] = {1, 1, 2, 3, 5, 8, 13, 21};

extern "C" __global__ void poly_eval(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = ((coeffs[3] * v + coeffs[2]) * v + coeffs[1]) * v + coeffs[0] + (float)lut[i & 7];
    atomicAdd(&counter, 1);
}
