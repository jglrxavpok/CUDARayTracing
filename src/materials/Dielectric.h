//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once
#include "Material.h"

class Dielectric: public Material {
private:
    double refractiveIndex;

public:
    explicit Dielectric(double refractiveIndex);

    bool scatter(const Ray &ray, const HitResult &hit, Color &attenuation, Ray &scattered) const override;
};
