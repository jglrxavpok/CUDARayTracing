//
// Created by jglrxavpok on 04/09/2020.
//

#include "Metal.h"

__host__ __device__ Metal::Metal(const Color &a, double fuzzyness): albedo(a), fuzzyness(fuzzyness < 1 ? fuzzyness : 1) {}

__host__ __device__ bool Metal::scatter(const Ray &ray, const HitResult &hit, Color &attenuation, Ray &scattered) const {
    Vec3 reflected = reflect(ray.direction().normalized(), hit.normal.normalized());
    scattered = Ray(hit.point, reflected/* TODO + fuzzyness * Vec3::randomInUnitSphere()*/);
    attenuation = albedo;
    return dot(scattered.direction(), hit.normal) > 0;
}
