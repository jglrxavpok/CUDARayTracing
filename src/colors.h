//
// Created by jglrxavpok on 03/09/2020.
//

#pragma once

#include "Vec3.h"

__host__ __device__ inline double clamp(double d, double min, double max) {
    if(d < min) return min;
    if(d > max) return max;
    return d;
}

__host__ __device__ inline void writeColor(uint8_t* pixels, size_t startPointer, const Color color, int samplesPerPixel) {
    auto scale = 1.0/samplesPerPixel;

    // gamma correction + average sampling
    double r = sqrt(scale*color.x());
    double g = sqrt(scale*color.y());
    double b = sqrt(scale*color.z());

    pixels[startPointer+0] = static_cast<int>(255*clamp(r, 0, 0.9999)); // blue
    pixels[startPointer+1] = static_cast<int>(255*clamp(g, 0, 0.9999)); // green
    pixels[startPointer+2] = static_cast<int>(255*clamp(b, 0, 0.9999)); // blue
    pixels[startPointer+3] = 255; // alpha
}