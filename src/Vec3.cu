//
// Created by jglrxavpok on 04/09/2020.
//
#include <curand_kernel.h>
#include "Vec3.h"
#include "rt.h"

__host__ __device__ Vec3 Vec3::random(curandState* rand, double min, double max) {
    return { randomDouble(rand, min, max), randomDouble(rand, min, max), randomDouble(rand, min, max) };
}

__host__ __device__ Vec3 Vec3::randomInUnitSphere(curandState* rand) {
    Vec3 v;
    do {
        v = random(rand, -1.0, 1.0);
    } while(v.lengthSquared() >= 1.0);
    return v;
}

__host__ __device__ Vec3 Vec3::randomUnitVector(curandState* rand) {
    auto a = randomDouble(rand, 0, 2*PI);
    auto z = randomDouble(rand, -1, 1);
    auto r = sqrt(1.0 - z*z);
    return {r*cos(a), r*sin(a), z};
}


