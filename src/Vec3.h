//
// Created by jglrxavpok on 03/09/2020.
//

#pragma once

#include "math.h"
#include <iostream>
#include <curand_kernel.h>

class Vec3 {
private:
    double _x;
    double _y;
    double _z;

public:
    __host__ __device__ Vec3(): _x(0), _y(0), _z(0) {}
    __host__ __device__ Vec3(double x, double y, double z): _x(x), _y(y), _z(z) {}

    __host__ __device__ double x() const {
        return this->_x;
    }

    __host__ __device__ double y() const {
        return this->_y;
    }

    __host__ __device__ double z() const {
        return this->_z;
    }

    __host__ __device__ Vec3 operator-() const {
        return {-x(), -y(), -z()};
    }

    __host__ __device__ Vec3& operator-=(const Vec3& other) {
        _x -= other.x();
        _y -= other.y();
        _z -= other.z();
        return *this;
    }

    __host__ __device__ Vec3& operator+=(const Vec3& other) {
        _x += other.x();
        _y += other.y();
        _z += other.z();
        return *this;
    }

    __host__ __device__ double lengthSquared() const {
        return _x*_x+_y*_y+_z*_z;
    }

    __host__ __device__ double length() const {
        return sqrt(lengthSquared());
    }

    __host__ __device__ Vec3 normalized() const {
        double l = length();
        return { x()/l, y()/l, z()/l };
    }

public:
    __host__ __device__ static Vec3 random(curandState* rand, double min, double max);
    __host__ __device__ static Vec3 randomInUnitSphere(curandState* rand);
    __host__ __device__ static Vec3 randomUnitVector(curandState* rand);
};

inline std::ostream& operator<<(std::ostream &out, const Vec3 &v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

__host__ __device__ inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x()+b.x(), a.y()+b.y(), a.z()+b.z()};
}

__host__ __device__ inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x()-b.x(), a.y()-b.y(), a.z()-b.z()};
}

__host__ __device__ inline Vec3 operator*(const Vec3& a, const Vec3& b) {
    return {a.x()*b.x(), a.y()*b.y(), a.z()*b.z()};
}

__host__ __device__ inline Vec3 operator*(const Vec3& a, const double t) {
    return {a.x()*t, a.y()*t, a.z()*t};
}

__host__ __device__ inline Vec3 operator*(const double t, const Vec3& a) {
    return a*t;
}

__host__ __device__ inline Vec3 operator/(const Vec3& a, const double t) {
    return {a.x()/t, a.y()/t, a.z()/t};
}

__host__ __device__ inline Vec3 operator/(const double t, const Vec3& a) {
    return a/t;
}

__host__ __device__ inline double dot(const Vec3& a, const Vec3& b) {
    return a.x()*b.x() + a.y()*b.y() + a.z()*b.z();
}

__host__ __device__ inline static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y()*b.z()-a.z()*b.y(),
        a.z()*b.x()-a.x()*b.z(),
        a.x()*b.y()-a.y()*b.x(),
    };
}

__host__ __device__ inline static Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2*dot(v,n)*n;
}

__host__ __device__ inline static Vec3 refract(const Vec3& v, const Vec3& n, double refractiveIndexRatio) {
    Vec3 perpendicularComponent = refractiveIndexRatio * (v + dot(-v, n) * n);
    Vec3 parallelComponent = -sqrt(fabs(1-perpendicularComponent.lengthSquared())) * n;
    return perpendicularComponent+parallelComponent;
}

using Point3 = Vec3;
using Color = Vec3;