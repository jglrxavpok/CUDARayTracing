//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once

#include "Intersectable.h"
#include <memory>
#include <vector>
#include <thrust/device_vector.h>

using thrust::device_vector;
using std::shared_ptr;

class IntersectableGroup: public Intersectable {
private:
    int elementCount;
    Intersectable** elements;
    AABB aabb;

public:
    __device__ IntersectableGroup(int elementCount, Intersectable** elements);
    __device__ bool hit(const Ray &ray, double mint, double maxt, HitResult &result) const override;

    __device__ const AABB &getAABB() const override;

private:
    __device__ static AABB wrap(int elementCount, Intersectable** elements);
};
