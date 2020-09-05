//
// Created by jglrxavpok on 04/09/2020.
//

#include "Sphere.h"

#include <utility>
#include "Material.h"

__device__ Sphere::Sphere(Point3 center, double radius, Material* material): center(center), radius(radius), material(material) {}

__device__ void Sphere::fillResult(HitResult& result, const Ray& ray, double t) const {
    Point3 intersectionPoint = ray.at(t);
    Vec3 normal = (intersectionPoint-getCenter()) / getRadius();
    result.t = t;
    result.normal = normal;
    result.point = intersectionPoint;
    result.material = material;
}

__device__ bool Sphere::hit(const Ray &ray, double mint, double maxt, HitResult &result) const {
    auto ac = ray.origin()-getCenter();
    auto a = ray.direction().lengthSquared();
    auto halfB = dot(ac, ray.direction());
    auto c = ac.lengthSquared()-getRadius()*getRadius();
    auto discriminant = halfB * halfB - a * c;
    if(discriminant < 0.0) {
        return false;
    } else {
        double root = sqrt(discriminant);
        double solution1 = (-halfB - root) / a;
        if(solution1 > mint && solution1 < maxt) {
            fillResult(result, ray, solution1);
            return true;
        }

        double solution2 = (-halfB + root) / a;
        if(solution2 > mint && solution2 < maxt) {
            fillResult(result, ray, solution2);
            return true;
        }

        return false;
    }
}

__device__ Point3 Sphere::getCenter() const {
    return center;
}

__device__ double Sphere::getRadius() const {
    return radius;
}
