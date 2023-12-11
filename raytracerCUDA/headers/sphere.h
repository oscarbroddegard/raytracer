#pragma once

#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "ray.h"
#include "hitable.h"

class sphere: public hitable {
public:
	vec3 center;
	float radius;
	material sphere_material;

	__device__ sphere() {}
	__device__ sphere(const vec3& c, const float& r,const material& m) :center(c), radius(r), sphere_material(m) {}

	__device__ virtual bool hit(const ray& r,float tmin,float tmax, intersection& isect);
};

#endif