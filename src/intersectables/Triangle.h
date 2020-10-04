//
// Created by jglrxavpok on 05/09/2020.
//

#pragma once

#include "Intersectable.h"
#include "Vec3.h"

class Triangle: public Intersectable {
public:
    __device__ explicit Triangle(
            Point3 a, Point3 b, Point3 c,
            Vec3 normalA, Vec3 normalB, Vec3 normalC,
            Vec3 uvA, Vec3 uvB, Vec3 uvC,

            Material* material);

    __device__ bool hit(const Ray &ray, double mint, double maxt, HitResult &result) const override;

    __device__ const AABB &getAABB() const override;

private:
    AABB aabb;
    Point3 a;
    Point3 b;
    Point3 c;
    Vec3 normalA;
    Vec3 normalB;
    Vec3 normalC;
    Vec3 uvA;
    Vec3 uvB;
    Vec3 uvC;
    Material* material;

    __device__ static Vec3 minVec(Vec3 a, Vec3 b, Vec3 c);
    __device__ static Vec3 maxVec(Vec3 a, Vec3 b, Vec3 c);
};
