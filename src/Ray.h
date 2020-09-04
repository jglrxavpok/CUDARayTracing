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
    explicit Ray();
    explicit Ray(Vec3 origin, Vec3 direction);

    Vec3 at(double t) const;

    Vec3 origin() const;
    Vec3 direction() const;
};
