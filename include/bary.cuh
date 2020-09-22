#ifndef BARY_H_
#define BARY_H_

#include "geometry.cuh"

namespace bary{
    bool* point_in_simplex(vec3f *pts, int n, int dim, vec3f *verts);
    float** bary_triangle(vec3f *pts, int n, int dim, vec3f *verts);
}

#endif //BARY_H_