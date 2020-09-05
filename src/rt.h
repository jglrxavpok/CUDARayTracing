//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once

#include "Vec3.h"
#include "Ray.h"
#include "Intersectable.h"
#include "intersectables/IntersectableGroup.h"
#include "intersectables/Sphere.h"
#include "colors.h"

#include <random>
#include <curand_kernel.h>

constexpr double PI = 3.1415926535897932385;

__device__ inline double randomDouble(curandState* rand) {
#ifdef __CUDA_ARCH__
    return curand_uniform_double(rand);
#else
    return 0.5;
#endif
}

__device__ inline double randomDouble(curandState* rand, double min, double max) {
    return randomDouble(rand)*(max-min) + min;
}