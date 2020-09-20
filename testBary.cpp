#include <iostream>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include "bary.cuh"

#define N 1000

int main(){
  srand(time(NULL));

  vec3f *pts_h;
  pts_h = new vec3f[N];

  // randomly generate N points
  for(int i=0; i<N; ++i){
    pts_h[i] = (vec3f){rand()%255, rand()%255, 0};
  }

  // pts_h[0] = {5,5,5};
  // vec3f t0 = {0,0,0};
  // vec3f t1 = {0, 100, 0};
  // vec3f t2 = {100, 0, 0};

  vec3f t0 = {rand()%255, rand()%255, 0};
  vec3f t1 = {rand()%255, rand()%255, 0};
  vec3f t2 = {rand()%255, rand()%255, 0};

  bool* res = bary::point_in_triangle(pts_h, N, t0, t1, t2);

  for(int i=N; i--; std::cout << res[i]){}

  return 0;
}