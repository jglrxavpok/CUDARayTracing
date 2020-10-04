//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once
#include "Intersectable.h"

class Sphere: public Intersectable {
public:
    __device__ Sphere(Point3 center, double radius, Material* material);

    __device__ Point3 getCenter() const;
    __device__ double getRadius() const;
    __device__ bool hit(const Ray &ray, double mint, double maxt, HitResult &result) const override;

    __device__ const AABB& getAABB() const override;

private:
    AABB aabb;
    Point3 center;
    double radius;
    Material* material;

    __device__ void fillResult(HitResult& result, const Ray& ray, double t) const;
};
