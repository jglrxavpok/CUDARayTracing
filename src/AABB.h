//
// Created by jglrxavpok on 04/10/2020.
//

#pragma once

#include "Vec3.h"
#include "Ray.h"

class AABB {
private:
    Vec3 min;
    Vec3 max;

public:
    __device__ AABB(Vec3 min, Vec3 max): min(min), max(max) {}
    __device__ bool intersects(const Ray& ray) const;

    __device__ const Vec3& getMin() const;
    __device__ const Vec3& getMax() const;

private:
    __device__ static void correctMinMax(float& a, float& b);
};
