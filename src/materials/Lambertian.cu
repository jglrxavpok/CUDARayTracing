//
// Created by jglrxavpok on 04/09/2020.
//

#include "Lambertian.h"

__device__ Lambertian::Lambertian(const Color &a): albedo(a) {}

__device__ bool
Lambertian::scatter(const Ray &ray, const HitResult &hit, curandState *rand, Color &attenuation, Ray &scattered) const {
    Vec3 direction = hit.normal + Vec3::randomInUnitSphere(rand);
    scattered = Ray(hit.point, direction);
    attenuation = albedo;
    return true;
}
