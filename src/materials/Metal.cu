//
// Created by jglrxavpok on 04/09/2020.
//

#include "Metal.h"

Metal::Metal(const Color &a, double fuzzyness): albedo(a), fuzzyness(fuzzyness < 1 ? fuzzyness : 1) {}

bool Metal::scatter(const Ray &ray, const HitResult &hit, Color &attenuation, Ray &scattered) const {
    Vec3 reflected = reflect(ray.direction().normalized(), hit.normal.normalized());
    scattered = Ray(hit.point, reflected + fuzzyness * Vec3::randomInUnitSphere());
    attenuation = albedo;
    return dot(scattered.direction(), hit.normal) > 0;
}
