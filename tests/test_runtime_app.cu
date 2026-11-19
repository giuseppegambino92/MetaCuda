// test_runtime_app.cu — an ordinary CUDA Runtime API program.
//
// Nothing here knows about MetaCuda: it uses <<<...>>> launches, templates,
// __constant__ / __device__ symbols, managed memory, streams, events and the
// usual error-checking idioms. It is compiled by clang exactly the way clang's
// CUDA driver compiles any .cu file (device pass → PTX → fatbin, host pass
// embeds the fatbin), links against libcudart, and runs on the Apple GPU.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                                                    \
    do {                                                                                                    \
        cudaError_t _e = (expr);                                                                            \
        if (_e != cudaSuccess) {                                                                            \
            std::fprintf(stderr, "CUDA error %s (%s) at %s:%d\n", cudaGetErrorName(_e), cudaGetErrorString(_e), \
                         __FILE__, __LINE__);                                                               \
            std::exit(1);                                                                                   \
        }                                                                                                   \
    } while (0)

static int failures = 0;
#define REQUIRE(cond, ...)                          \
    do {                                            \
        if (!(cond)) {                              \
            std::fprintf(stderr, "  FAILED: " __VA_ARGS__); \
            std::fprintf(stderr, "\n");             \
            failures++;                             \
        }                                           \
    } while (0)

__constant__ float c_transform[2];       // scale, offset
__device__ unsigned long long d_launches; // incremented by every block of `tally`

// Compute-heavy kernel with data-dependent loops.
__global__ void mandelbrot(unsigned short *iters, int w, int h, int maxit, float x0, float y0, float dx) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    float cr = x0 + px * dx, ci = y0 + py * dx;
    float zr = 0.f, zi = 0.f;
    int it = 0;
    while (it < maxit && zr * zr + zi * zi <= 4.0f) {
        float t = zr * zr - zi * zi + cr;
        zi = 2.0f * zr * zi + ci;
        zr = t;
        ++it;
    }
    iters[py * w + px] = (unsigned short)it;
}

// Template kernel: two instantiations, two mangled PTX entries.
template <typename T>
__global__ void axpb(const T *x, T *y, T a, T b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b;
}

// Row-wise softmax: one block per row, dynamic shared memory reductions.
__global__ void softmax_rows(const float *in, float *out, int cols) {
    extern __shared__ float red[];
    const float *row = in + (size_t)blockIdx.x * cols;
    float *orow = out + (size_t)blockIdx.x * cols;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, row[c]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += expf(row[c] - m);
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int c = threadIdx.x; c < cols; c += blockDim.x) orow[c] = expf(row[c] - m) * inv;
}

__global__ void apply_constant(float *v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = v[i] * c_transform[0] + c_transform[1];
}

__global__ void tally() {
    if (threadIdx.x == 0) atomicAdd(&d_launches, 1ull);
}

// CPU reference with exactly the operation sequence of the compiler-generated PTX
// (fma.rn.f32 where the PTX has it, separate mul/sub/add elsewhere), with host-side
// contraction disabled — so the GPU result must match bit for bit.
static int mandel_ref(int px, int py, int maxit, float x0, float y0, float dx) {
#pragma clang fp contract(off)
    float cr = std::fmaf((float)px, dx, x0), ci = std::fmaf((float)py, dx, y0), zr = 0, zi = 0;
    int k = 0;
    while (k < maxit && std::fmaf(zr, zr, zi * zi) <= 4.0f) {
        float t = (zr * zr - zi * zi) + cr;
        zi = std::fmaf(zr + zr, zi, ci);
        zr = t;
        ++k;
    }
    return k;
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    std::printf("CUDA runtime %d / driver %d, %d device(s)\n", rt, drv, count);
    std::printf("device 0: %s, sm_%d%d, %d SMs, %.1f GiB, %zu B shared/block, warp %d, unified addressing %d, concurrent kernels %d\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0, prop.sharedMemPerBlock,
                prop.warpSize, prop.unifiedAddressing, prop.concurrentKernels);
    REQUIRE(prop.concurrentKernels == 0 && prop.asyncEngineCount == 0 && prop.deviceOverlap == 0, "device must not claim concurrency it lacks");
    CUDA_CHECK(cudaSetDevice(0));

    // ---- mandelbrot ----
    {
        const int w = 1024, h = 768, maxit = 256;
        const float x0 = -2.2f, y0 = -1.2f, dx = 3.2f / w;
        unsigned short *d_it;
        CUDA_CHECK(cudaMalloc(&d_it, w * h * sizeof(unsigned short)));
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
        mandelbrot<<<grid, block>>>(d_it, w, h, 1, x0, y0, dx); // warm-up: first launch JIT-compiles the module (lazy loading)
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t0));
        mandelbrot<<<grid, block>>>(d_it, w, h, maxit, x0, y0, dx);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        std::vector<unsigned short> it(w * h);
        CUDA_CHECK(cudaMemcpy(it.data(), d_it, it.size() * sizeof(unsigned short), cudaMemcpyDeviceToHost));
        long mismatches = 0, total = 0;
        for (int py = 0; py < h; py++)
            for (int px = 0; px < w; px++) {
                int k = mandel_ref(px, py, maxit, x0, y0, dx);
                total += k;
                if (it[py * w + px] != k) mismatches++;
            }
        std::printf("mandelbrot %dx%d: %.3f ms on GPU, %.0f Miter/s, %ld/%d pixels differ from the bit-exact CPU reference\n", w, h, ms,
                    total / (ms * 1e3), mismatches, w * h);
        REQUIRE(mismatches == 0, "mandelbrot mismatches %ld", mismatches);
        CUDA_CHECK(cudaFree(d_it));
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    // ---- template kernels on a non-default stream ----
    {
        const int n = 100000;
        std::vector<float> xf(n);
        std::vector<int> xi(n);
        for (int i = 0; i < n; i++) { xf[i] = 0.001f * i; xi[i] = i - n / 2; }
        float *dxf, *dyf;
        int *dxi, *dyi;
        CUDA_CHECK(cudaMalloc(&dxf, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dyf, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dxi, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dyi, n * sizeof(int)));
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));
        CUDA_CHECK(cudaMemcpyAsync(dxf, xf.data(), n * sizeof(float), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(dxi, xi.data(), n * sizeof(int), cudaMemcpyHostToDevice, s));
        axpb<float><<<(n + 255) / 256, 256, 0, s>>>(dxf, dyf, 3.0f, -1.0f, n);
        axpb<int><<<(n + 255) / 256, 256, 0, s>>>(dxi, dyi, -7, 11, n);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> yf(n);
        std::vector<int> yi(n);
        CUDA_CHECK(cudaMemcpyAsync(yf.data(), dyf, n * sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(yi.data(), dyi, n * sizeof(int), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        int bad = 0;
        for (int i = 0; i < n; i++) {
            if (std::fabs(yf[i] - (3.0f * xf[i] - 1.0f)) > 1e-4f) bad++;
            if (yi[i] != -7 * xi[i] + 11) bad++;
        }
        std::printf("axpb<float>, axpb<int> on stream: %d errors\n", bad);
        REQUIRE(bad == 0, "template kernels");
        cudaStreamDestroy(s);
        cudaFree(dxf); cudaFree(dyf); cudaFree(dxi); cudaFree(dyi);
    }

    // ---- softmax with dynamic shared memory ----
    {
        const int rows = 512, cols = 1000;
        std::vector<float> in(rows * cols), out(rows * cols);
        for (int i = 0; i < rows * cols; i++) in[i] = std::sin(0.37f * i) * 8.0f;
        float *din, *dout;
        CUDA_CHECK(cudaMalloc(&din, in.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dout, out.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(din, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
        const int threads = 256;
        softmax_rows<<<rows, threads, threads * sizeof(float)>>>(din, dout, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double maxerr = 0, maxrowsum = 0;
        for (int r = 0; r < rows; r++) {
            float m = -INFINITY;
            for (int c = 0; c < cols; c++) m = std::fmax(m, in[r * cols + c]);
            double s = 0, rs = 0;
            for (int c = 0; c < cols; c++) s += std::exp((double)in[r * cols + c] - m);
            for (int c = 0; c < cols; c++) {
                double ref = std::exp((double)in[r * cols + c] - m) / s;
                maxerr = std::fmax(maxerr, std::fabs(ref - out[r * cols + c]));
                rs += out[r * cols + c];
            }
            maxrowsum = std::fmax(maxrowsum, std::fabs(rs - 1.0));
        }
        std::printf("softmax %dx%d: max abs error %.2e, max |rowsum-1| %.2e\n", rows, cols, maxerr, maxrowsum);
        REQUIRE(maxerr < 1e-5 && maxrowsum < 1e-4, "softmax accuracy");
        cudaFree(din); cudaFree(dout);
    }

    // ---- __constant__ symbol, managed memory, __device__ counter ----
    {
        const int n = 4096;
        float *v;
        CUDA_CHECK(cudaMallocManaged(&v, n * sizeof(float)));
        for (int i = 0; i < n; i++) v[i] = (float)i; // host writes managed memory directly
        float tr[2] = {0.5f, 100.0f};
        CUDA_CHECK(cudaMemcpyToSymbol(c_transform, tr, sizeof tr));
        apply_constant<<<n / 128, 128>>>(v, n);
        unsigned long long zero = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(d_launches, &zero, sizeof zero));
        for (int k = 0; k < 50; k++) tally<<<20, 64>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        int bad = 0;
        for (int i = 0; i < n; i++) if (v[i] != 0.5f * i + 100.0f) bad++; // host reads managed memory directly
        unsigned long long launches = 0;
        CUDA_CHECK(cudaMemcpyFromSymbol(&launches, d_launches, sizeof launches));
        void *sym = nullptr;
        CUDA_CHECK(cudaGetSymbolAddress(&sym, d_launches));
        std::printf("__constant__ + managed memory: %d errors; __device__ counter = %llu (expect 1000) at %p\n", bad, launches, sym);
        REQUIRE(bad == 0 && launches == 1000, "symbols/managed");
        cudaPointerAttributes pa;
        CUDA_CHECK(cudaPointerGetAttributes(&pa, v));
        REQUIRE(pa.type == cudaMemoryTypeManaged, "managed pointer attributes");
        CUDA_CHECK(cudaFree(v));
    }

    // ---- error handling idioms ----
    {
        tally<<<1, 4096>>>(); // too many threads per block
        cudaError_t e = cudaGetLastError();
        std::printf("invalid launch -> %s (\"%s\")\n", cudaGetErrorName(e), cudaGetErrorString(e));
        REQUIRE(e == cudaErrorInvalidConfiguration, "invalid configuration must be reported");
        REQUIRE(cudaGetLastError() == cudaSuccess, "error must be cleared");
        cudaFuncAttributes fa;
        CUDA_CHECK(cudaFuncGetAttributes(&fa, softmax_rows));
        std::printf("cudaFuncGetAttributes(softmax_rows): maxThreadsPerBlock=%d maxDynamicShared=%d\n", fa.maxThreadsPerBlock, fa.maxDynamicSharedSizeBytes);
        REQUIRE(fa.maxThreadsPerBlock > 0, "func attributes");
        size_t freeb = 0, totalb = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
        REQUIRE(totalb > 0 && freeb <= totalb, "mem info");
    }

    std::printf(failures ? "test_runtime_app: FAILED (%d)\n" : "test_runtime_app: PASSED\n", failures);
    return failures ? 1 : 0;
}
