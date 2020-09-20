/** 
  barycuda - a CUDA library for fast barycentric operations
  
  Copyright (c) 2020 Srimukh Sripada
  MIT License
*/

#include <cstdio>
#include "bary.cuh"

#define N_THREADS_PER_BLOCK 512

#define cudaErrchk(ans) { gpuSay((ans), __FILE__, __LINE__);}

inline void gpuSay(cudaError_t scode, const char *file, int line, bool abort=true){
  if (scode != cudaSuccess){
    fprintf(stderr, "gpuSay: %s %s %d\n", cudaGetErrorString(scode), file, line);
    if (abort)
      exit(scode);
  }
}


namespace bary{
  __global__ void point_in_triangle_kernel(vec3f *pts_d, 
                                           vec3f *t0_d, 
                                           vec3f *t1_d, 
                                           vec3f *t2_d, 
                                           bool *res_d){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    vec3f v0 = {t1_d->x - t0_d->x, t1_d->y - t0_d->y, t1_d->z - t0_d->z};
    vec3f v1 = {t2_d->x - t0_d->x, t2_d->y - t0_d->y, t2_d->z - t0_d->z};
    vec3f v2 = {pts_d[i].x - t0_d->x, pts_d[i].y - t0_d->y, pts_d[i].z - t0_d->z};
    float deno = v0.x * v1.y - v1.x * v0.y;
    float v = (v2.x * v1.y - v1.x * v2.y) / deno;
    float w = (v0.x * v2.y - v2.x * v0.y) / deno;
    float u = 1 - (v + w);
    bool r = !(u < 0 || v < 0 || w < 0);
    res_d[i] = r;
  }

  // Checks if a bunch of points lie inside a triangle
  // @param pts An array of points
  // @param n number of points
  // @param t0 vertex of triangle
  // @param t1 vertex of triangle
  // @param t2 vertex of triangle
  // @return res a boolean array
  __host__ bool* point_in_triangle(vec3f *pts, 
                                   int n, 
                                   vec3f t0, 
                                   vec3f t1, 
                                   vec3f t2) {
    vec3f *pts_d;
    vec3f *t0_d, *t1_d, *t2_d;
    bool *res_d;
    bool *res = new bool[n];

    cudaErrchk(cudaMalloc(&pts_d, n * sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&t0_d, sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&t1_d, sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&t2_d, sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&res_d, n * sizeof(bool)));

    cudaErrchk(cudaMemcpy(pts_d, pts, n * sizeof(vec3f), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(t0_d, &t0, sizeof(vec3f), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(t1_d, &t1, sizeof(vec3f), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(t2_d, &t2, sizeof(vec3f), cudaMemcpyHostToDevice));

    point_in_triangle_kernel<<<1+(n/N_THREADS_PER_BLOCK), N_THREADS_PER_BLOCK>>>(pts_d, t0_d, t1_d, t2_d, res_d);

    cudaErrchk(cudaMemcpy(res, res_d, n * sizeof(bool), cudaMemcpyDeviceToHost));

    cudaFree(pts_d);
    cudaFree(t0_d);
    cudaFree(t1_d);
    cudaFree(t2_d);
    cudaFree(res_d);
    return res;
  }
}