//
// Created by jglrxavpok on 04/09/2020.
//

#include "Ray.h"

__host__ __device__ Ray::Ray(): _origin(Vec3()), _direction(Vec3()) {}
__host__ __device__ Ray::Ray(Vec3 origin, Vec3 direction): _origin(origin), _direction(direction) {}

__host__ __device__ Vec3 Ray::at(double t) const {
    return _origin + _direction*t;
}

__host__ __device__ Vec3 Ray::origin() const {
    return _origin;
}

__host__ __device__ Vec3 Ray::direction() const {
    return _direction;
}