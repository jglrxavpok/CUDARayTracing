cmake_minimum_required(VERSION 3.16)
project(CUDARayTracing CUDA CXX)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)

set(CXXFILES
        src/main.cu
        src/Ray.cpp
        src/collisions.cpp
        src/Intersectable.cpp
        src/intersectables/Sphere.cpp
        src/intersectables/IntersectableGroup.cpp
        src/Camera.cpp
        src/Vec3.cpp
        src/Material.cpp
        src/materials/Lambertian.cpp
        src/materials/Metal.cpp
        src/materials/Dielectric.cpp
        )

include_directories(${PROJECT_SOURCE_DIR}/src)
add_executable(CUDARayTracing ${CXXFILES})
target_compile_options(CUDARayTracing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)

set_target_properties(
        CUDARayTracing
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)