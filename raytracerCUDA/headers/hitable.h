#pragma once

#ifndef HITABLE_H
#define HITABLE_H

#include "device_launch_parameters.h"
#include "color.h"
#include "vec3.h"
#include "ray.h"
#include "material.h"

struct intersection {
	double hit_t;
	vec3 hit_position;
	vec3 hit_normal;
	material hit_material;
};

class hitable {
public:
	virtual bool hit(const ray& r, double tmin, double tmax, intersection& isect) = 0;
};

#endif