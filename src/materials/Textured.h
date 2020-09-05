//
// Created by jglrxavpok on 05/09/2020.
//

#pragma once

#include "Material.h"
#include "Texture.h"

class Textured: public Material {
public:
    __device__ explicit Textured(Texture* texture, Material* baseMaterial, double blendFactor = 0.5);
    __device__ explicit Textured(Texture* texture);

    __device__ bool scatter(const Ray &ray, const HitResult &hit, curandState *rand, Color &attenuation, Ray &scattered) const override;

private:
    Texture* texture;
    Material* baseMaterial;
    double blendFactor;
};

