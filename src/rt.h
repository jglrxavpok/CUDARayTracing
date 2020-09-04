//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once

#include "Vec3.h"
#include "Ray.h"
#include "Intersectable.h"
#include "intersectables/IntersectableGroup.h"
#include "intersectables/Sphere.h"
#include "colors.h"

#include <random>

constexpr double PI = 3.1415926535897932385;

inline double randomDouble() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double randomDouble(double min, double max) {
    return randomDouble()*(max-min) + min;
}