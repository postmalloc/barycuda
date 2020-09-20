cmake_minimum_required(VERSION 3.0)
project(barycuda)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30)

cuda_add_library(barycuda SHARED bary.cu)

cuda_add_executable(test_bary testBary.cpp)
target_link_libraries(test_bary barycuda)
