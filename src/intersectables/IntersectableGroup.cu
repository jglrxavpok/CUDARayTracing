//
// Created by jglrxavpok on 04/09/2020.
//

#include "IntersectableGroup.h"

IntersectableGroup::IntersectableGroup(vector<shared_ptr<Intersectable>> &&v): elements(std::move(v)) {}
IntersectableGroup::IntersectableGroup(std::initializer_list<shared_ptr<Intersectable>> l): elements(l) {}

bool IntersectableGroup::hit(const Ray &ray, double mint, double maxt, HitResult &result) const {
    auto closest = maxt;
    HitResult tmpResult{};
    auto hit = false;

    for(const auto& element : elements) {
        if(element->hit(ray, mint, closest, tmpResult)) {
            closest = tmpResult.t;
            result = tmpResult;
            hit = true;
        }
    }
    return hit;
}
