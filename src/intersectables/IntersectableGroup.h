//
// Created by jglrxavpok on 04/09/2020.
//

#pragma once

#include "Intersectable.h"
#include <memory>
#include <vector>

using std::vector;
using std::shared_ptr;

class IntersectableGroup: public Intersectable {
private:
    vector<shared_ptr<Intersectable>> elements;

public:
    IntersectableGroup(std::initializer_list<shared_ptr<Intersectable>> l);
    explicit IntersectableGroup(vector<shared_ptr<Intersectable>>&& v);
    bool hit(const Ray &ray, double mint, double maxt, HitResult &result) const override;
};
