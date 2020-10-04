//
// Created by jglrxavpok on 04/09/2020.
//

#include "IntersectableGroup.h"

__device__ IntersectableGroup::IntersectableGroup(int elementCount, Intersectable** elements): elementCount(elementCount), elements(elements),
aabb(wrap(elementCount, elements)) {}

__device__ bool IntersectableGroup::hit(const Ray &ray, double mint, double maxt, HitResult &result) const {
    auto closest = maxt;
    HitResult tmpResult{};
    auto hit = false;

    for(int i = 0; i < elementCount; i++) {
        if(elements[i]->trace(ray, mint, closest, tmpResult)) {
            closest = tmpResult.t;
            result = tmpResult;
            hit = true;
        }
    }
    return hit;
}

__device__ const AABB &IntersectableGroup::getAABB() const {
    return aabb;
}

__device__ AABB IntersectableGroup::wrap(int elementCount, Intersectable** elements) {
    Vec3 minVec{INFINITY, INFINITY, INFINITY};
    Vec3 maxVec{-INFINITY, -INFINITY, -INFINITY};
    for (int i = 0; i < elementCount; i++) {
        Intersectable* element = elements[i];
        const AABB& elementBounds = element->getAABB();
        minVec.setX(min(minVec.x(), elementBounds.getMin().x()));
        minVec.setY(min(minVec.y(), elementBounds.getMin().y()));
        minVec.setZ(min(minVec.z(), elementBounds.getMin().z()));

        maxVec.setX(max(maxVec.x(), elementBounds.getMax().x()));
        maxVec.setY(max(maxVec.y(), elementBounds.getMax().y()));
        maxVec.setZ(max(maxVec.z(), elementBounds.getMax().z()));
    }
    return {minVec, maxVec};
}
