//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once
#include "Vec3.h"
#include "HitResult.h"
#include "Ray.h"

class Intersectable {
public:
    __device__ virtual bool hit(const Ray& ray, double mint, double maxt, HitResult& result) const = 0;
    virtual ~Intersectable() = default;
};
