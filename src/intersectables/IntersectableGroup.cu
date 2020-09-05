//
// Created by jglrxavpok on 04/09/2020.
//

#include "IntersectableGroup.h"

__host__ __device__ IntersectableGroup::IntersectableGroup(int elementCount, Intersectable** elements): elementCount(elementCount), elements(elements) {}

__host__ __device__ bool IntersectableGroup::hit(const Ray &ray, double mint, double maxt, HitResult &result) const {
    auto closest = maxt;
    HitResult tmpResult{};
    auto hit = false;

    for(int i = 0; i < elementCount; i++) {
        if(elements[i]->hit(ray, mint, closest, tmpResult)) {
            closest = tmpResult.t;
            result = tmpResult;
            hit = true;
        }
    }
    return hit;
}
