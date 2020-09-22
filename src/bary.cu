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
    *res = (vec3f){v2->x-v1->x, v2->y-v1->y, v2->z-v1->z};
  }

  __device__ void _cross(vec3f *res, vec3f *v1, vec3f *v2){
    res->x = v1->y*v2->z - v1->z*v2->y;
    res->y = v1->z*v2->x - v1->x*v2->z;
    res->z = v1->x*v2->y - v1->y*v2->x;
  }

  __device__ float _dot(vec3f *v1, vec3f *v2){
    return v1->x*v2->x + v1->y*v2->y + v1->z*v2->z;
  }

  __device__ bool _all(float *bary, int len){
    for(int i=0; i<len; i++){
      if(bary[i] < 0) return false;
    }
    return true;
  }

  __global__ void pt_in_tri(vec3f *pts, int npts, vec3f *a, vec3f *b, vec3f *c, bool *res){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = max(i, 0);
    i = min(i, npts);

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

    res[i] = _all(bary, 3);
  }

  __global__ void pt_in_tet(vec3f *pts, int npts, vec3f *a, vec3f *b, vec3f *c, vec3f *d, bool *res){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = max(i, 0);
    i = min(i, npts);

    vec3f *p = &pts[i];
    vec3f ap[3], bp[3];
    vec3f ab[3], ac[3], ad[3];
    vec3f bc[3], bd[3], tmp[3];
    float _a, _b, _c, _d, _v;
    float bary[4];

    _diff(ap, a, p);
    _diff(bp, b, p);
    _diff(ab, a, b);
    _diff(ac, a, c);
    _diff(ad, a, d);
    _diff(bc, b, c);
    _diff(bd, b, d);

    // Compute the scalar triple products
    _cross(tmp, bd, bc);
    _a = _dot(bp, tmp);
    _cross(tmp, ac, ad);
    _b = _dot(ap, tmp);
    _cross(tmp, ad, ab);
    _c = _dot(ap, tmp);
    _cross(tmp, ab, ac);
    _d = _dot(ap, tmp);
    _cross(tmp, ac, ad);
    _v = abs(_dot(ab, tmp));

    // Calculate the relative volumes of the subtetrahedrons
    bary[0] = _a / _v;
    bary[1] = _b / _v;
    bary[2] = _c / _v;
    bary[3] = _d / _v;

    res[i] = _all(bary, 4);

  }


  /**
    Checks if a bunch of points lie inside a 2D/3D simplex
    @param pts An array of points
    @param n number of simplex vertices
    @param verts array of simplex vertices
    @return res a boolean array
  */
  __host__ bool* point_in_simplex(vec3f *pts, int n, int dim, vec3f *verts) {
    vec3f *pts_d;
    vec3f *a_d, *b_d, *c_d, *d_d;
    bool *res_d;
    bool *res = new bool[n];

    cudaErrchk(cudaMalloc(&pts_d, n * sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&a_d, sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&b_d, sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&c_d, sizeof(vec3f)));
    cudaErrchk(cudaMalloc(&res_d, n * sizeof(bool)));

    cudaErrchk(cudaMemcpy(pts_d, pts, n * sizeof(vec3f), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(a_d, &verts[0], sizeof(vec3f), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(b_d, &verts[1], sizeof(vec3f), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(c_d, &verts[2], sizeof(vec3f), cudaMemcpyHostToDevice));

    // Check if the simplex is 2D or 3D
    if(dim == 4){
      cudaErrchk(cudaMalloc(&d_d, sizeof(vec3f)));
      cudaErrchk(cudaMemcpy(d_d, &verts[3], sizeof(vec3f), cudaMemcpyHostToDevice));
      pt_in_tet<<<1+(n/N_THREADS_PER_BLOCK), N_THREADS_PER_BLOCK>>>(pts_d, n, a_d, b_d, c_d, d_d, res_d);
    } else{
      pt_in_tri<<<1+(n/N_THREADS_PER_BLOCK), N_THREADS_PER_BLOCK>>>(pts_d, n, a_d, b_d, c_d, res_d);
    }

    cudaErrchk(cudaMemcpy(res, res_d, n * sizeof(bool), cudaMemcpyDeviceToHost));

    cudaFree(pts_d);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFree(d_d);
    cudaFree(res_d);
    return res;
  }

}