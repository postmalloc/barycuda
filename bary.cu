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

  __global__ void pt_in_tri(vec3f *pts, vec3f *a, vec3f *b, vec3f *c, bool *res){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    vec3f *p = &pts[i];
    vec3f ap[3], bp[3], cp[3];
    vec3f ac[3], ab[3], ca[3], bc[3];
    vec3f n[3], n0[3], n1[3], n2[3];

    float bary[3];

    // Vectors from vertices to the point
    _diff(ap, a, p);
    _diff(bp, b, p);
    _diff(cp, c, p);

    // Vectors for triangle edges
    _diff(ac, a, c);
    _diff(ab, a, b);
    _diff(ca, c, a);
    _diff(bc, b, c);

    // Make normals. The lengths are proportional to the
    // area of the subtriangles
    _cross(n, ab, ac);
    _cross(n0, bc, bp);
    _cross(n1, ca, cp);
    _cross(n2, ab, ap);

    float n_norm = _dot(n, n);

    // Find relative areas
    bary[0] = _dot(n, n0) / n_norm;
    bary[1] = _dot(n, n1) / n_norm;
    bary[2] = _dot(n, n2) / n_norm;

    res[i] = !(bary[0]<0 || bary[1]<0 || bary[2]<0);
  }


  /**
    Checks if a bunch of points lie inside a triangle
    @param pts An array of points
    @param n number of points
    @param a vertex of triangle
    @param b vertex of triangle
    @param c vertex of triangle
    @return res a boolean array
  */
  __host__ bool* point_in_triangle(vec3f *pts, int n, vec3f a, vec3f b, vec3f c) {
    vec3f *pts_d;
    vec3f *a_d, *b_d, *c_d;
    bool *res_d;
    bool *res = new bool[n];

    cudaErrchk(cudaMalloc(&pts_d, n * sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&a_d, sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&b_d, sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&c_d, sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&res_d, n * sizeof(bool)));

    cudaErrchk(cudaMemcpy(pts_d, pts, n * sizeof(vec3f), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(a_d, &a, sizeof(vec3f), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(b_d, &b, sizeof(vec3f), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(c_d, &c, sizeof(vec3f), cudaMemcpyHostToDevice));

    pt_in_tri<<<1+(n/N_THREADS_PER_BLOCK), N_THREADS_PER_BLOCK>>>(pts_d, a_d, b_d, c_d, res_d);

    cudaErrchk(cudaMemcpy(res, res_d, n * sizeof(bool), cudaMemcpyDeviceToHost));

    cudaFree(pts_d);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(res_d);
    return res;
  }
}