//
// Created by jglrxavpok on 05/09/2020.
//

#include "Triangle.h"

__device__ Triangle::Triangle(Point3 a, Point3 b, Point3 c,
        Vec3 normalA, Vec3 normalB, Vec3 normalC,
        Vec3 uvA, Vec3 uvB, Vec3 uvC,
        Material *material): a(a), b(b), c(c), normalA(normalA), normalB(normalB), normalC(normalC),
        uvA(uvA), uvB(uvB), uvC(uvC),
        material(material),
        aabb(minVec(a, b, c), maxVec(a, b, c)){}

// Möller–Trumbore algorithm
__device__ bool Triangle::hit(const Ray &ray, double mint, double maxt, HitResult &result) const {
    // TODO
    const float EPSILON = 0.000001;
    Point3 vertex0 = this->a;
    Point3 vertex1 = this->b;
    Point3 vertex2 = this->c;
    Vec3 edge1 = vertex1-vertex0;
    Vec3 edge2 = vertex2-vertex0;
    Vec3 h = cross(ray.direction(), edge2);
    float a = dot(edge1, h);
    if(a > -EPSILON && a < EPSILON) {
        return false; // ray parallel to triangle
    }

    float f = 1.0f/a;
    Vec3 s = ray.origin() - vertex0;
    float u = f * dot(s, h);
    if(u < 0.0 || u > 1.0)
        return false;
    Vec3 q = cross(s, edge1);
    float v = f * dot(ray.direction(), q);
    if(v < 0.0 || u+v > 1.0)
        return false;

    float t = f * dot(edge2, q);
    if(t > EPSILON && t > mint && t < maxt) {
        result.t = t;
        result.material = material;
        result.point = ray.at(t);
        float w = 1-v-u;
        result.normal = normalA * w + normalB * u + normalC * v;
        result.uvwMapping = uvA * w + uvB * u + uvC * v;
        return true;
    }
    return false;
}

__device__ const AABB &Triangle::getAABB() const {
    return aabb;
}

__device__ Vec3 Triangle::maxVec(Vec3 a, Vec3 b, Vec3 c) {
    double maxX = max(max(a.x(), b.x()), c.x());
    double maxY = max(max(a.y(), b.y()), c.y());
    double maxZ = max(max(a.z(), b.z()), c.z());
    return {maxX, maxY, maxZ};
}

__device__ Vec3 Triangle::minVec(Vec3 a, Vec3 b, Vec3 c) {
    double minX = min(min(a.x(), b.x()), c.x());
    double minY = min(min(a.y(), b.y()), c.y());
    double minZ = min(min(a.z(), b.z()), c.z());
    return {minX, minY, minZ};
}