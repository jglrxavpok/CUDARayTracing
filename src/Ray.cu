//
// Created by jglrxavpok on 04/09/2020.
//

#include "Ray.h"

__device__ Ray::Ray(): _origin(Vec3()), _direction(Vec3()) {}
__device__ Ray::Ray(Vec3 origin, Vec3 direction): _origin(origin), _direction(direction) {}

__device__ Vec3 Ray::at(double t) const {
    return _origin + _direction*t;
}

__device__ Vec3 Ray::origin() const {
    return _origin;
}

__device__ Vec3 Ray::direction() const {
    return _direction;
}