cmake_minimum_required(VERSION 3.10)

project(readMapper-build)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(CUDA 10.0 REQUIRED)

project(xact-build LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -std=c++17 -Wall -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

find_package(Boost COMPONENTS program_options REQUIRED)

find_package(ZLIB REQUIRED)
find_path(ZLIB_INCLUDE_DIR zlib.h)
include_directories(${ZLIB_INCLUDE_DIR})

include(${TBB_DIR}/cmake/TBBBuild.cmake)
tbb_build(TBB_ROOT ${TBB_DIR} CONFIG_DIR TBB_DIR MAKE_ARGS tbb_cpf=1)
find_package(TBB REQUIRED tbbmalloc tbbmalloc_proxy tbb_preview)

add_executable (readMapper
    src/readMapper.cu
    src/twoBitCompressor.cpp
    src/main.cpp)

target_link_libraries(readMapper PRIVATE ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} ${TBB_IMPORTED_TARGETS})

