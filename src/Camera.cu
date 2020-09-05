//
// Created by jglrxavpok on 04/09/2020.
//

#include "Camera.h"

__device__ Camera::Camera(Point3 lookFrom, Point3 lookAt, Vec3 vup, double fovy, double aspectRatio): fovy(fovy), aspectRatio(aspectRatio) {
    double theta = fovy * PI / 180.0;
    double h = tan(theta/2);
    double viewportHeight = 2.0 * h;
    double viewportWidth = viewportHeight * aspectRatio;

    auto w = (lookFrom-lookAt).normalized();
    auto u = cross(vup, w).normalized();
    auto v = cross(w, u);

    origin = lookFrom;
    horizontal = viewportWidth*u;
    vertical = viewportHeight*v;
    lowerLeftCorner = origin - horizontal / 2.0 - vertical / 2.0 - w;
}

__device__ Ray Camera::generateRay(double u, double v) const {
    return Ray(origin, lowerLeftCorner+u*horizontal+v*vertical - origin);
}
