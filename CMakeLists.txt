cmake_minimum_required(VERSION 3.0)
project(barycuda)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30)
include_directories(./include)

cuda_add_library(barycuda SHARED src/bary.cu)

cuda_add_executable(test_bary src/testBary.cpp)
target_link_libraries(test_bary barycuda)
