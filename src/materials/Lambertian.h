//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once

#include "Material.h"

class Lambertian: public Material {
private:
    Color albedo;

public:
    explicit Lambertian(const Color& a);

    bool scatter(const Ray &ray, const HitResult &hit, Color &attenuation, Ray &scattered) const override;
};
