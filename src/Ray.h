//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once

#include "Vec3.h"

class Ray {
private:
    Vec3 _origin;
    Vec3 _direction;

public:
    __device__ explicit Ray();
    __device__ explicit Ray(Vec3 origin, Vec3 direction);

    __device__ Vec3 at(double t) const;

    __device__ Vec3 origin() const;
    __device__ Vec3 direction() const;
};
