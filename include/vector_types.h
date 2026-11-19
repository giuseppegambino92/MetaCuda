/* MetaCuda — CUDA built-in vector types (subset of NVIDIA's vector_types.h). */
#ifndef METACUDA_VECTOR_TYPES_H
#define METACUDA_VECTOR_TYPES_H

#if defined(__CUDA__) && !defined(__host__)
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#endif
#if !defined(__CUDA__)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

#define MC_VEC2(T, N) typedef struct __attribute__((aligned(sizeof(T) * 2))) N##2 { T x, y; } N##2;
#define MC_VEC3(T, N) typedef struct N##3 { T x, y, z; } N##3;
#define MC_VEC4(T, N) typedef struct __attribute__((aligned(sizeof(T) * 4))) N##4 { T x, y, z, w; } N##4;
#define MC_VECS(T, N) MC_VEC2(T, N) MC_VEC3(T, N) MC_VEC4(T, N)

MC_VECS(float, float)
MC_VECS(int, int)
MC_VECS(unsigned int, uint)
MC_VECS(unsigned char, uchar)
MC_VECS(char, char)
MC_VECS(short, short)
MC_VECS(unsigned short, ushort)

#ifdef __cplusplus
struct dim3 {
    unsigned int x, y, z;
    __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __host__ __device__ constexpr operator uint3() const { return uint3{x, y, z}; }
};
#else
typedef struct dim3 { unsigned int x, y, z; } dim3;
#endif
typedef struct dim3 dim3;

#endif
