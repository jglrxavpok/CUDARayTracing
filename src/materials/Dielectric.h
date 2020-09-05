//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once
#include "Material.h"

class Dielectric: public Material {
private:
    double refractiveIndex;

public:
    __host__ __device__ explicit Dielectric(double refractiveIndex);

    __host__ __device__ bool scatter(const Ray &ray, const HitResult &hit, Color &attenuation, Ray &scattered) const override;
};
