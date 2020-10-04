//
// Created by jglrxavpok on 04/10/2020.
//

#include "AABB.h"

/// Swaps expectedMin and expectedMax if expectedMin > expectedMax
__device__ void AABB::correctMinMax(float& expectedMin, float& expectedMax) {
    if(expectedMin > expectedMax) { // swap values if a > b
        float tmp = expectedMin;
        expectedMin = expectedMax;
        expectedMax = tmp;
    }
}

__device__ bool AABB::intersects(const Ray &ray) const {
    float invX = 1.0f / ray.direction().x();
    float invY = 1.0f / ray.direction().y();
    float invZ = 1.0f / ray.direction().z();
    float t0x = (min.x() - ray.origin().x()) * invX;
    float t0y = (min.y() - ray.origin().y()) * invY;
    float t0z = (min.z() - ray.origin().z()) * invZ;

    float t1x = (max.x() - ray.origin().x()) * invX;
    float t1y = (max.y() - ray.origin().y()) * invY;
    float t1z = (max.z() - ray.origin().z()) * invZ;

    correctMinMax(t0x, t1x);
    correctMinMax(t0y, t1y);
    correctMinMax(t0z, t1z);

    float tmin = (t0x > t0y) ? t0x : t0y;
    float tmax = (t1x < t1y) ? t1x : t1y;
    if(t0x > t1y || t0y > t1x) return false; // out of box
    if(tmin > t1z || t0z > tmax) // out of box
        return false;
    if(t0z > tmin)
        tmin = t0z;
    if(t1z < tmax)
        tmax = t1z;

    return true;
}

__host__ __device__ const Vec3 &AABB::getMin() const {
    return min;
}

__host__ __device__ const Vec3 &AABB::getMax() const {
    return max;
}
