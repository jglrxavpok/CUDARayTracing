//
// Created by jglrxavpok on 04/09/2020.
//

#include "Lambertian.h"

Lambertian::Lambertian(const Color &a): albedo(a) {}

bool Lambertian::scatter(const Ray &ray, const HitResult &hit, Color &attenuation, Ray &scattered) const {
    Vec3 direction = hit.normal + Vec3::randomUnitVector();
    scattered = Ray(hit.point, direction);
    attenuation = albedo;
    return true;
}
