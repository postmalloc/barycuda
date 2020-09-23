#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <time.h>
#include "bary.cuh"

#define N 1

int main(){
  srand(time(NULL));

  vec3f *pts_in = new vec3f[N];
  vec3f *pts_out = new vec3f[N];

  // Define vertices of a tetrahedron
  vec3f verts[] = {
    {0,    0,   0}, 
    {100,  0,   0}, 
    {0,  100,   0}, 
    {0,    0, 100}
  };


  // =============== Point inside the tetrahedron ==============
  pts_in[0] = (vec3f){20,10,5};
  bool* pts_in_res = bary::point_in_simplex(pts_in, N, 4, verts);
  float** bary_in = bary::bary_simplex(pts_in, N, 4, verts);

  // The point should lie inside the tetrahedron
  assert(pts_in_res[0] == true);

  // Test for the barycentric coordinates
  assert(abs(bary_in[0][0] - 0.65) < 0.001);
  assert(abs(bary_in[0][1] - 0.20) < 0.001);
  assert(abs(bary_in[0][2] - 0.10) < 0.001);
  assert(abs(bary_in[0][3] - 0.05) < 0.001);


  // ============== Point outside the tetrahedron ==============
  pts_out[0] = (vec3f){120,50,90};
  bool* pts_out_res = bary::point_in_simplex(pts_out, N, 4, verts);
  float** bary_out = bary::bary_simplex(pts_out, N, 4, verts);

  // The point should be inside the tetrahedron
  assert(pts_out_res[0] == false);

  // Test for the barycentric coordinates
  assert(abs(bary_out[0][0] - (-1.60)) < 0.001);
  assert(abs(bary_out[0][1] - 1.20) < 0.001);
  assert(abs(bary_out[0][2] - 0.50) < 0.001);
  assert(abs(bary_out[0][3] - 0.90) < 0.001);

  return 0;
}