#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

struct intersection
{
    float hit_t;
    vec3 hit_position;
    vec3 hit_normal;
    //material* hit_material;
};

class hitable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, intersection& rec) const = 0;
};

#endif