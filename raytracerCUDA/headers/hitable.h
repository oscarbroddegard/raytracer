#pragma once

#ifndef HITABLE_H
#define HITABLE_H

#include "device_launch_parameters.h"
#include "color.h"
#include "vec3.h"
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
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, intersection& isect) = 0;
};
#endif