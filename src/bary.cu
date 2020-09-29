/**
  barycuda - a CUDA library for fast barycentric operations

  Copyright (c) 2020 Srimukh Sripada
  MIT License
*/

#include <cstdio>
#include "bary.cuh"

#define N_THREADS_PER_BLOCK 64
#define BLOCK_X 3
#define BLOCK_Y 32

#define gpuErrChk(ans) \
  { gpuSay((ans), __FILE__, __LINE__); }

inline void gpuSay(cudaError_t scode, const char *file, int line,
                   bool abort = true) {
  if (scode != cudaSuccess) {
    fprintf(stderr, "gpuSay: %s %s %d\n", cudaGetErrorString(scode), file,
            line);
    if (abort) exit(scode);
  }
}

namespace bary {

__device__ void _diff(vec3f *res, vec3f *v1, vec3f *v2) {
  *res = (vec3f){v2->x - v1->x, v2->y - v1->y, v2->z - v1->z};
}

__device__ void _cross(vec3f *res, vec3f *v1, vec3f *v2) {
  res->x = v1->y * v2->z - v1->z * v2->y;
  res->y = v1->z * v2->x - v1->x * v2->z;
  res->z = v1->x * v2->y - v1->y * v2->x;
}

__device__ float _dot(vec3f *v1, vec3f *v2) {
  return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}

__device__ bool _all(float *bary, int len) {
  for (int i = 0; i < len; i++) {
    if (bary[i] < 0) return false;
  }
  return true;
}

__host__ int _ceilDiv(int n, int k) {
  return ((n % k) != 0) ? (n / k + 1) : (n / k);
}

__device__ void _bary_tri(vec3f *p, int n, vec3f *a, vec3f *b, vec3f *c,
                          float *res) {
  vec3f ap[3], bp[3], cp[3];
  vec3f ac[3], ab[3], ca[3], bc[3];
  vec3f nor[3], nor0[3], nor1[3], nor2[3];

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
  _cross(nor, ab, ac);
  _cross(nor0, bc, bp);
  _cross(nor1, ca, cp);
  _cross(nor2, ab, ap);

  float n_norm = _dot(nor, nor);

  // Find relative areas
  res[0] = _dot(nor, nor0) / n_norm;
  res[1] = _dot(nor, nor1) / n_norm;
  res[2] = _dot(nor, nor2) / n_norm;
}

__device__ void _bary_tet(vec3f *p, int n, vec3f *a, vec3f *b, vec3f *c,
                          vec3f *d, float *res) {
  vec3f ap[3], bp[3];
  vec3f ab[3], ac[3], ad[3];
  vec3f bc[3], bd[3], tmp[3];
  float _a, _b, _c, _d, _v;

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
  res[0] = _a / _v;
  res[1] = _b / _v;
  res[2] = _c / _v;
  res[3] = _d / _v;
}

__global__ void bary_tri(vec3f *pts, int n, vec3f *a, vec3f *b, vec3f *c,
                         size_t pitch, float *res) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;

  if ((tidx < 3) && (tidy < n)) {
    vec3f *p = &pts[tidy];
    float bary[3];
    _bary_tri(p, n, a, b, c, bary);
    float *row = (float *)((char *)res + tidy*pitch);
    row[0] = bary[0];
    row[1] = bary[1];
    row[2] = bary[2];
  }
}

__global__ void bary_tet(vec3f *pts, int n, vec3f *a, vec3f *b, vec3f *c,
                         vec3f *d, size_t pitch, float *res) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;

  if ((tidx < 4) && (tidy < n)) {
    vec3f *p = &pts[tidy];
    float bary[4];
    _bary_tet(p, n, a, b, c, d, bary);
    float *row = (float *)((char *)res + tidy*pitch);
    row[0] = bary[0];
    row[1] = bary[1];
    row[2] = bary[2];
    row[3] = bary[3];
  }
}

__global__ void pt_in_tri(vec3f *pts, int n, vec3f *a, vec3f *b, vec3f *c,
                          bool *res) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  i = max(i, 0);
  i = min(i, n);
  vec3f *p = &pts[i];
  float bary[3];
  _bary_tri(p, n, a, b, c, bary);
  res[i] = _all(bary, 3);
}

__global__ void pt_in_tet(vec3f *pts, int n, vec3f *a, vec3f *b, vec3f *c,
                          vec3f *d, bool *res) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  i = max(i, 0);
  i = min(i, n);
  vec3f *p = &pts[i];
  float bary[4];
  _bary_tet(p, n, a, b, c, d, bary);
  res[i] = _all(bary, 4);
}

__host__ float **bary_simplex(vec3f *pts, int n, int ndim, vec3f *verts) {
  vec3f *pts_d;
  vec3f *a_d, *b_d, *c_d, *d_d;
  float *res_d;
  float *res = new float[n * ndim];
  float **res2d = new float *[n];
  size_t pitch;
  for (int i = 0; i < n; i++) {
    res2d[i] = &res[i * ndim];
  }

  gpuErrChk(cudaMalloc(&pts_d, n * sizeof(vec3f)));
  gpuErrChk(cudaMalloc(&a_d, sizeof(vec3f)));
  gpuErrChk(cudaMalloc(&b_d, sizeof(vec3f)));
  gpuErrChk(cudaMalloc(&c_d, sizeof(vec3f)));

  gpuErrChk(cudaMallocPitch((void **)&res_d, &pitch, ndim * sizeof(float), n));
  gpuErrChk(cudaMemcpy2D(res_d, pitch, res, ndim * sizeof(float),
                         ndim * sizeof(float), n, cudaMemcpyHostToDevice));

  gpuErrChk(cudaMemcpy(pts_d, pts, n * sizeof(vec3f), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(a_d, &verts[0], sizeof(vec3f), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(b_d, &verts[1], sizeof(vec3f), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(c_d, &verts[2], sizeof(vec3f), cudaMemcpyHostToDevice));

  dim3 gridSize(_ceilDiv(ndim, BLOCK_X), _ceilDiv(n, BLOCK_Y));
  dim3 blockSize(BLOCK_Y, BLOCK_X);

  if (ndim == 4) {
    gpuErrChk(cudaMalloc(&d_d, sizeof(vec3f)));
    gpuErrChk(cudaMemcpy(d_d, &verts[3], sizeof(vec3f),
                         cudaMemcpyHostToDevice));
    bary_tet<<<gridSize, blockSize>>>(pts_d, n, a_d, b_d, c_d, d_d, pitch,
                                      res_d);
  } else {
    bary_tri<<<gridSize, blockSize>>>(pts_d, n, a_d, b_d, c_d, pitch, res_d);
  }

  gpuErrChk(cudaMemcpy2D(res, ndim * sizeof(float), res_d, pitch,
                         ndim * sizeof(float), n, cudaMemcpyDeviceToHost));
  
  cudaFree(pts_d);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_d);
  cudaFree(res_d);

  return res2d;
}

/**
  Checks if a bunch of points lie inside a 2D/3D simplex
  @param pts An array of points
  @param n number of simplex vertices
  @param verts array of simplex vertices
  @return res a boolean array
*/
__host__ bool *point_in_simplex(vec3f *pts, int n, int ndim, vec3f *verts) {
  vec3f *pts_d;
  vec3f *a_d, *b_d, *c_d, *d_d;
  bool *res_d;
  bool *res = new bool[n];

  gpuErrChk(cudaMalloc(&pts_d, n * sizeof(vec3f)));
  gpuErrChk(cudaMalloc(&a_d, sizeof(vec3f)));
  gpuErrChk(cudaMalloc(&b_d, sizeof(vec3f)));
  gpuErrChk(cudaMalloc(&c_d, sizeof(vec3f)));
  gpuErrChk(cudaMalloc(&res_d, n * sizeof(bool)));

  gpuErrChk(cudaMemcpy(pts_d, pts, n * sizeof(vec3f), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(a_d, &verts[0], sizeof(vec3f), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(b_d, &verts[1], sizeof(vec3f), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(c_d, &verts[2], sizeof(vec3f), cudaMemcpyHostToDevice));

  // Check if the simplex is 2D or 3D
  if (ndim == 4) {
    gpuErrChk(cudaMalloc(&d_d, sizeof(vec3f)));
    gpuErrChk(cudaMemcpy(d_d, &verts[3], sizeof(vec3f),
                         cudaMemcpyHostToDevice));
    pt_in_tet<<<_ceilDiv(n, N_THREADS_PER_BLOCK), N_THREADS_PER_BLOCK>>>(
        pts_d, n, a_d, b_d, c_d, d_d, res_d);
  } else {
    pt_in_tri<<<_ceilDiv(n, N_THREADS_PER_BLOCK), N_THREADS_PER_BLOCK>>>(
        pts_d, n, a_d, b_d, c_d, res_d);
  }

  gpuErrChk(cudaMemcpy(res, res_d, n * sizeof(bool), cudaMemcpyDeviceToHost));

  cudaFree(pts_d);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_d);
  cudaFree(res_d);
  return res;
}
}