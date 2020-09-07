//
// Created by jglrxavpok on 07/09/2020.
//

#pragma once

#include "Intersectable.h"
#include "Triangle.h"
#include "IntersectableGroup.h"

class TriangleMesh: public Intersectable {
private:
    int triangleCount;
    Triangle** triangles;
    Material* material;

public:
    __host__ static TriangleMesh** loadFromFile(const std::string& name);
};
