//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once
#include "Intersectable.h"

class Sphere: public Intersectable {
public:
    Sphere(Point3 center, double radius, shared_ptr<Material> material);

    Point3 getCenter() const;
    double getRadius() const;
    shared_ptr<Material>& getMaterial();
    const shared_ptr<Material>& getMaterial() const;
    bool hit(const Ray &ray, double mint, double maxt, HitResult &result) const override;

private:
    Point3 center;
    double radius;
    shared_ptr<Material> material;

    void fillResult(HitResult& result, const Ray& ray, double t) const;
};
