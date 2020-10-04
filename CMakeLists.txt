cmake_minimum_required(VERSION 3.16)

project(CUDARayTracing CUDA CXX)

find_package(CUDA)
set(assimp_DIR "vcpkg/packages/assimp_x64-windows/share/assimp/")
find_package(assimp CONFIG REQUIRED)
set(CMAKE_CUDA_STANDARD 14)


set(CXXFILES
        src/main.cu
        src/Ray.cu
        src/HitResult.cu
        src/Intersectable.cu
        src/intersectables/Sphere.cu
        src/intersectables/Triangle.cu
        src/intersectables/TriangleMesh.cu
        src/intersectables/IntersectableGroup.cu
        src/Camera.cu
        src/Vec3.cu
        src/Material.cu
        src/materials/Lambertian.cu
        src/materials/Metal.cu
        src/materials/Dielectric.cu
        src/materials/Textured.cu
        src/Texture.cu
        src/AABB.cu
        )

include_directories(${PROJECT_SOURCE_DIR}/src)
add_executable(CUDARayTracing ${CXXFILES})
#GTX950M
#target_compile_options(CUDARayTracing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)
#GTX970
target_compile_options(CUDARayTracing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_52,code=sm_52>)

target_link_libraries(CUDARayTracing PRIVATE assimp::assimp)

set_target_properties(
        CUDARayTracing
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)