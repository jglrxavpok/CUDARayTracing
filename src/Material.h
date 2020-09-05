//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once
#include "rt.h"

class HitResult;

class Material {
public:
    __host__ __device__ virtual bool scatter(const Ray& ray, const HitResult& hit, Color& attenuation, Ray& scattered) const = 0;

    virtual ~Material() = default;
};
