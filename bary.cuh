#ifndef BARY_H_
#define BARY_H_

#include "geometry.cuh"

namespace bary{
    bool* point_in_triangle(vec3f *pts, int n, vec3f t0, vec3f t1, vec3f t2);
}

#endif //BARY_H_