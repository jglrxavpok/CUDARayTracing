//
// Created by jglrxavpok on 05/09/2020.
//

#pragma once

#include "Vec3.h"

class Texture {
public:
    __device__ Texture(int width, int height, Color* pixels);
    virtual ~Texture() = default;

    __device__ virtual Color at(Point3 position);

    __host__ static Texture** loadFromFile(const std::string& name);

private:
    int width;
    int height;
    Color* pixels;

};
