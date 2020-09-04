//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once
#include "Vec3.h"
#include "Ray.h"

using std::shared_ptr;

class Material;

class HitResult {
public:
    Point3 point;
    Vec3 normal;
    double t;
    shared_ptr<Material> material;
};
