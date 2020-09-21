#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "bary.cuh"

#define N 1000

int main(){
  srand(time(NULL));

  vec3f *pts_h;
  pts_h = new vec3f[N];

  // randomly generate N points
  for(int i=0; i<N; ++i){
    pts_h[i] = (vec3f){rand()%255, rand()%255, rand()%255};
  }

  // pts_h[0] = (vec3f){100,10,5};
  // vec3f t0 = {0,0,0};
  // vec3f t1 = {100, 0, 0};
  // vec3f t2 = {0, 100, 0};
  // vec3f t3 = {0, 0, 100};

  vec3f t0 = {rand()%255, rand()%255, rand()%255};
  vec3f t1 = {rand()%255, rand()%255, rand()%255};
  vec3f t2 = {rand()%255, rand()%255, rand()%255};
  vec3f t3 = {rand()%255, rand()%255, rand()%255};

  vec3f *verts = new vec3f[4];
  verts[0] = t0;
  verts[1] = t1;
  verts[2] = t2;
  verts[3] = t3;

  bool* res = bary::point_in_simplex(pts_h, N, 4, verts);

  for(int i=N; i--; std::cout << res[i]){}

  return 0;
}