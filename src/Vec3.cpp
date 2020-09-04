//
// Created by jglrxavpok on 04/09/2020.
//
#include "Vec3.h"
#include "rt.h"

Vec3 Vec3::random() {
    return { randomDouble(), randomDouble(), randomDouble() };
}

Vec3 Vec3::random(double min, double max) {
    return { randomDouble(min, max), randomDouble(min, max), randomDouble(min, max) };
}

Vec3 Vec3::randomInUnitSphere() {
    Vec3 v;
    do {
        v = random(-1.0, 1.0);
    } while(v.lengthSquared() >= 1.0);
    return v;
}

Vec3 Vec3::randomUnitVector() {
    auto a = randomDouble(0, 2*PI);
    auto z = randomDouble(-1, 1);
    auto r = sqrt(1.0 - z*z);
    return {r*cos(a), r*sin(a), z};
}


