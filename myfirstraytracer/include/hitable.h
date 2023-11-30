#pragma once

#ifndef HITABLE_H
#define HITABLE_H

#include "color.h"
#include "vec3.h"
#include "ray.h"

struct material {
	color diffusecolor;
};

struct intersection {
	double hit_t;
	vec3 hit_position;
	vec3 hit_normal;
	material hit_material;
};

class hitable {
public:
	virtual bool hit(const ray& r, double tmin, double tmax, intersection& isect) const = 0;
private:
	material material;
};

#endif