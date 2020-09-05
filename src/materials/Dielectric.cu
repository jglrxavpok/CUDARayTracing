//
// Created by jglrxavpok on 04/09/2020.
//

#include "Dielectric.h"

__host__ __device__ Dielectric::Dielectric(double refractiveIndex): refractiveIndex(refractiveIndex) {}

__host__ __device__ double schlick(double cosine, double refractiveIndex) {
    auto r0 = (1-refractiveIndex) / (1+refractiveIndex);
    r0 = r0*r0;
    return r0+(1-r0)*pow((1-cosine),5);
}

__host__ __device__ bool
Dielectric::scatter(const Ray &ray, const HitResult &hit, curandState *rand, Color &attenuation, Ray &scattered) const {
    attenuation = Color(1,1,1);
    double refractiveIndexRatio = refractiveIndex;
    Vec3 normal = hit.normal.normalized();
    if(dot(hit.normal, ray.direction()) < 0) { // entering material
        refractiveIndexRatio = 1.0/refractiveIndexRatio;
    } else {
        normal = -normal;
    }

    Vec3 incidentRay = ray.direction().normalized();

    double cosTheta = dot(-incidentRay, normal);
    double cosTheta2 = fmin(1.0, cosTheta*cosTheta);
    double sinTheta = sqrt(1-cosTheta2);
    double reflectionProbability = schlick(cosTheta, refractiveIndexRatio); // glass acts like a mirror at grazing angles


    if(sinTheta*refractiveIndexRatio > 1.0 || randomDouble(rand) < reflectionProbability) {
        // reflection
        Vec3 reflected = reflect(incidentRay, normal);
        scattered = Ray(hit.point, reflected);
        return true;
    }

    Vec3 refracted = refract(incidentRay, normal, refractiveIndexRatio);
    scattered = Ray(hit.point, refracted);
    return true;
}
