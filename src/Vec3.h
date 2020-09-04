//
// Created by jglrxavpok on 03/09/2020.
//

#pragma once

#include "math.h"
#include <iostream>

class Vec3 {
private:
    double _x;
    double _y;
    double _z;

public:
    Vec3(): _x(0), _y(0), _z(0) {}
    Vec3(double x, double y, double z): _x(x), _y(y), _z(z) {}

    double x() const {
        return this->_x;
    }

    double y() const {
        return this->_y;
    }

    double z() const {
        return this->_z;
    }

    Vec3 operator-() const {
        return {-x(), -y(), -z()};
    }

    Vec3& operator-=(const Vec3& other) {
        _x -= other.x();
        _y -= other.y();
        _z -= other.z();
        return *this;
    }

    Vec3& operator+=(const Vec3& other) {
        _x += other.x();
        _y += other.y();
        _z += other.z();
        return *this;
    }

    double lengthSquared() const {
        return _x*_x+_y*_y+_z*_z;
    }

    double length() const {
        return sqrt(lengthSquared());
    }

    Vec3 normalized() const {
        double l = length();
        return { x()/l, y()/l, z()/l };
    }

public:
    static Vec3 random();
    static Vec3 random(double min, double max);
    static Vec3 randomInUnitSphere();
    static Vec3 randomUnitVector();
};

inline std::ostream& operator<<(std::ostream &out, const Vec3 &v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x()+b.x(), a.y()+b.y(), a.z()+b.z()};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x()-b.x(), a.y()-b.y(), a.z()-b.z()};
}

inline Vec3 operator*(const Vec3& a, const Vec3& b) {
    return {a.x()*b.x(), a.y()*b.y(), a.z()*b.z()};
}

inline Vec3 operator*(const Vec3& a, const double t) {
    return {a.x()*t, a.y()*t, a.z()*t};
}

inline Vec3 operator*(const double t, const Vec3& a) {
    return a*t;
}

inline Vec3 operator/(const Vec3& a, const double t) {
    return {a.x()/t, a.y()/t, a.z()/t};
}

inline Vec3 operator/(const double t, const Vec3& a) {
    return a/t;
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x()*b.x() + a.y()*b.y() + a.z()*b.z();
}

inline static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y()*b.z()-a.z()*b.y(),
        a.z()*b.x()-a.x()*b.z(),
        a.x()*b.y()-a.y()*b.x(),
    };
}

inline static Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2*dot(v,n)*n;
}

inline static Vec3 refract(const Vec3& v, const Vec3& n, double refractiveIndexRatio) {
    Vec3 perpendicularComponent = refractiveIndexRatio * (v + dot(-v, n) * n);
    Vec3 parallelComponent = -sqrt(fabs(1-perpendicularComponent.lengthSquared())) * n;
    return perpendicularComponent+parallelComponent;
}

using Point3 = Vec3;
using Color = Vec3;