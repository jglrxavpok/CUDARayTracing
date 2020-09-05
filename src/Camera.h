//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once
#include "rt.h"

class Camera {
public:
    static constexpr int SAMPLES_PER_PIXEL = 100;

    __host__ __device__ explicit Camera(Point3 lookFrom, Point3 lookAt, Vec3 vup, double fovy, double aspectRatio);

    __host__ __device__ Ray generateRay(double u, double v) const;

private:
    double fovy;
    double aspectRatio;
    Point3 origin;
    Point3 lowerLeftCorner;
    Vec3 vertical;
    Vec3 horizontal;
};
