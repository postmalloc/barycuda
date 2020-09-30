# barycuda
<img src="./docs/barycuda.png" alt="artwork" width="250"/>

Barycuda is a tiny CUDA accelerated library with no dependencies that
performs various barycentric operations. This is an attempt to speed-up
a 3D renderer that I'm working on.

Currently, the library exposes the following functions:  
* `point_in_simplex` - takes an array of points, the vertices
of a 2D/3D simplex, and tells you if each point lies inside the
simplex.  
* `bary_simplex` - takes an array of points, the vertices of
a 2D/3D simplex, and returns the barycentric coordinates for
each point.


## Usage
You can use Barycuda directly from C++ (see `src/testBary.cpp` for an example), 
or if you prefer using Python, Barycuda has a Python wrapper available. 
You can install it by doing  

`pip install pybarycuda`  

Installation using `pip` will come bundled with a prebuilt
`libbarycuda.so` for Linux x86-64. The linker paths are correctly set out of
the box, and you shouldn't have to change your `LD_LIBRARY_PATH` to be able to
use it.


I plan to support other architectures in the future, but if you want to
do it yourself, you can build the binaries (see below) and add `libbarycuda.so` 
to `LD_LIBRARY_PATH` to make the Python wrapper work.


You will need a CUDA capable system for any of this to work.

## Build
```bash
# inside the project directory
mkdir build
cd build
cmake ../
make
```

## Contributing
Feel free to add more geometric operations related to
graphics rendering and raytracing.

## References
Shirley, P. (2009) Fundamentals of Computer Graphics  
https://en.wikipedia.org/wiki/Barycentric_coordinate_system  
https://math.stackexchange.com/questions/1226707/how-to-check-if-point-x-in-mathbbrn-is-in-a-n-simplex  
