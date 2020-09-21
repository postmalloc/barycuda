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

  __device__ void _diff(vec3f *res, vec3f *v1, vec3f *v2){
    *res = (vec3f){v2->x-v1->x, v2->y-v1->y, v2->z-v2->z};
  }

  __device__ void _cross(vec3f *res, vec3f *v1, vec3f *v2){
    res->x = v1->y*v2->z - v1->z*v2->y;
    res->y = v1->z*v2->x - v1->x*v2->z;
    res->z = v1->x*v2->y - v1->y*v2->x;
  }

  __device__ float _dot(vec3f *v1, vec3f *v2){
    return v1->x*v2->x + v1->y*v2->y + v1->z*v2->z;
  }

  __global__ void point_in_triangle_kernel(vec3f *pts_d, 
                                           vec3f *t0_d, 
                                           vec3f *t1_d, 
                                           vec3f *t2_d, 
                                           bool *res_d){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    vec3f *p = &pts_d[i];
    vec3f t0p[3], t1p[3], t2p[3];
    vec3f t0t2[3], t0t1[3], t2t0[3], t1t2[3];
    vec3f n[3], n0[3], n1[3], n2[3];

    float bary[3];

    _diff(t0p, t0_d, p);
    _diff(t1p, t1_d, p);
    _diff(t2p, t2_d, p);

    _diff(t0t2, t0_d, t2_d);
    _diff(t0t1, t0_d, t1_d);
    _diff(t2t0, t2_d, t0_d);
    _diff(t1t2, t1_d, t2_d);

    _cross(n, t0t1, t0t2);
    _cross(n0, t1t2, t1p);
    _cross(n1, t2t0, t2p);
    _cross(n2, t0t1, t0p);

    float n_norm = _dot(n, n);

    bary[0] = _dot(n, n0) / n_norm;
    bary[1] = _dot(n, n1) / n_norm;
    bary[2] = _dot(n, n2) / n_norm;

    res_d[i] = !(bary[0]<0 || bary[1]<0 || bary[2]<0);
  }



  /**
    Checks if a bunch of points lie inside a triangle
    @param pts An array of points
    @param n number of points
    @param t0 vertex of triangle
    @param t1 vertex of triangle
    @param t2 vertex of triangle
    @return res a boolean array
  */
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