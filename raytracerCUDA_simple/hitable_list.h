#ifndef HITABLE_LIST_H
#define HITABLE_LIST_H

#include "hitable.h"

class hitable_list : public hitable {
public:
    __host__ __device__ hitable_list() {}
    __host__ __device__ hitable_list(hitable* l, int n) { list = l; n_hitables = n; }
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, intersection& rec) const;
    hitable* list;
    int n_hitables;
};

__device__ bool hitable_list::hit(const ray& r, float tmin, float tmax, intersection& isect) const {
    intersection temp;
    bool hit = false;
    float closest_t = tmax;
    for (int i = 0; i < n_hitables; i++) {
        if (list[i]->hit(r, tmin, closest_t, temp)) {
            hit = true;
            closest_t = temp.hit_t;
            isect = temp;
        }
    }
    return hit;
}

#endif