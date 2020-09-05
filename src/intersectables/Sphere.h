//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once
#include "Intersectable.h"

class Sphere: public Intersectable {
public:
    __host__ __device__ Sphere(Point3 center, double radius, Material* material);

    __host__ __device__ Point3 getCenter() const;
    __host__ __device__ double getRadius() const;
    __host__ __device__ bool hit(const Ray &ray, double mint, double maxt, HitResult &result) const override;

private:
    Point3 center;
    double radius;
    Material* material;

    __host__ __device__ void fillResult(HitResult& result, const Ray& ray, double t) const;
};
