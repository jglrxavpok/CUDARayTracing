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

private:
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
};
