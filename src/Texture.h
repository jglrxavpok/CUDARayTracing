//
// Created by jglrxavpok on 05/09/2020.
//

#pragma once

#include "Vec3.h"

class Texture {
public:
    __device__ virtual Color at(Point3 position);

    virtual ~Texture() = default;
};
