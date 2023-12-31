#pragma once

#ifndef HITABLE_H
#define HITABLE_H

#include "ray.h"
#include "material.h"


struct intersection {
	vec3 hit_position;
	float hit_t;
	vec3 hit_normal;
	material* hit_material;
};

class hitable {
public:
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, intersection& isect) const = 0;
};
#endif