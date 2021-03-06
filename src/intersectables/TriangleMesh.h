//
// Created by jglrxavpok on 07/09/2020.
//

#pragma once

#include "Intersectable.h"
#include "Triangle.h"
#include "IntersectableGroup.h"

// TODO: transforms
// TODO: optimisations
class TriangleMesh: public Intersectable {
private:
    IntersectableGroup* backingRepresentation;

public:
    __device__ explicit TriangleMesh(IntersectableGroup* backing);

    __device__ bool hit(const Ray &ray, double mint, double maxt, HitResult &result) const override;

    __device__ const AABB &getAABB() const override;

public:
    __host__ static TriangleMesh* loadFromFile(const std::string& name);
};
