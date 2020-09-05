//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once

#include <curand_kernel.h>
#include "Material.h"

class Dielectric: public Material {
private:
    double refractiveIndex;

public:
    __device__ explicit Dielectric(double refractiveIndex);

    __device__ bool
    scatter(const Ray &ray, const HitResult &hit, curandState *rand, Color &attenuation, Ray &scattered) const override;
};
