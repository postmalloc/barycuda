# barycuda
A tiny CUDA library for fast barycentric operations.

Barycuda is a tiny CUDA accelerated library with no dependencies that
performs various barycentric operations. I am building this on the side
to speed-up a 3D renderer I am working on. It currently only
supports batch point-in-polygon operations for triangles.

## Build
```bash
# inside the project directory
mkdir build
cd build
cmake ../
make
```

## Usage
Please see `testBary.cpp` to see how to use the library.

## Contributing
Feel free to add more geometric operations related to
graphics rendering and raytracing.