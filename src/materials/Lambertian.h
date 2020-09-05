//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once

#include <curand_kernel.h>
#include "Material.h"

class Lambertian: public Material {
private:
    Color albedo;

public:
    __host__ __device__ explicit Lambertian(const Color& a);

    __host__ __device__ bool
    scatter(const Ray &ray, const HitResult &hit, curandState *rand, Color &attenuation, Ray &scattered) const override;
};
