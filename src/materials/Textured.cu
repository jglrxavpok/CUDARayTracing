//
// Created by jglrxavpok on 05/09/2020.
//

#include "Textured.h"
#include "Lambertian.h"

__device__ Textured::Textured(Texture *texture, Material *baseMaterial): texture(texture), baseMaterial(baseMaterial) {}

__device__ Textured::Textured(Texture *texture): texture(texture) {
    baseMaterial = new Lambertian(Color(1,1,1));
}

__device__ bool Textured::scatter(const Ray &ray, const HitResult &hit, curandState *rand, Color &attenuation, Ray &scattered) const {
    bool result = baseMaterial->scatter(ray, hit, rand, attenuation, scattered);
    attenuation *= texture->at(hit.uvwMapping);
    return result;
}
