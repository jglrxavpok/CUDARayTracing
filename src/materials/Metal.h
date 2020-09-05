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
    __device__ explicit Metal(const Color& a, double fuzzyness);

    __device__ bool
    scatter(const Ray &ray, const HitResult &hit, curandState *rand, Color &attenuation, Ray &scattered) const override;
};
