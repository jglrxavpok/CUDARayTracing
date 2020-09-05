//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once

#include <curand_kernel.h>
#include "rt.h"

class HitResult;

class Material {
public:
    __device__ virtual bool scatter(const Ray &ray, const HitResult &hit, curandState *rand, Color &attenuation, Ray &scattered) const = 0;

    virtual ~Material() = default;
};
