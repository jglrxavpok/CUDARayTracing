//
// Created by jglrxavpok on 04/09/2020.
//

#include "Lambertian.h"

__host__ __device__ Lambertian::Lambertian(const Color &a): albedo(a) {}

__host__ __device__ bool Lambertian::scatter(const Ray &ray, const HitResult &hit, Color &attenuation, Ray &scattered) const {
    Vec3 direction = hit.normal;// TODO + Vec3::randomUnitVector();
    scattered = Ray(hit.point, direction);
    attenuation = albedo;
    return true;
}
