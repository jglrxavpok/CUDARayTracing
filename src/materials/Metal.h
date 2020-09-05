//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once

#include "Material.h"

class Metal: public Material {
private:
    Color albedo;
    double fuzzyness;

public:
    __host__ __device__ explicit Metal(const Color& a, double fuzzyness);

    __host__ __device__ bool scatter(const Ray &ray, const HitResult &hit, Color &attenuation, Ray &scattered) const override;
};
