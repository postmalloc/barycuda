# barycuda
A tiny CUDA library for fast barycentric operations.

Barycuda is a tiny CUDA accelerated library with no dependencies that
performs various barycentric operations. This is an attempt to speed-up
a 3D renderer that I'm working on.

Currently, the library provides functions to check if a set of points are inside a 2D/3D simplex. 

## Build
```bash
# inside the project directory
mkdir build
cd build
cmake ../
make
```

## Usage
Please see `testBary.cpp` 

## Contributing
Feel free to add more geometric operations related to
graphics rendering and raytracing.

## References
Shirley, P. (2009) Fundamentals of Computer Graphics  
https://en.wikipedia.org/wiki/Barycentric_coordinate_system  
https://math.stackexchange.com/questions/1226707/how-to-check-if-point-x-in-mathbbrn-is-in-a-n-simplex  
