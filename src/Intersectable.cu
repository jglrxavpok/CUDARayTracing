//
// Created by jglrxavpok on 04/09/2020.
//

#include "Intersectable.h"

__device__ bool Intersectable::trace(const Ray &ray, double mint, double maxt, HitResult &result) const {
    if(getAABB().intersects(ray)) {
        return hit(ray, mint, maxt, result);
    }
    return false;
}
