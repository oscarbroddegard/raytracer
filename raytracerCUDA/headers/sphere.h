#pragma once

#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "ray.h"
#include "hitable.h"

class sphere: public hitable {
public:
	vec3 center;
	double radius;
	material sphere_material;

	__host__ __device__ sphere() {}
	__host__ __device__ sphere(const vec3& c, const double& r,const material& m) :center(c), radius(r), sphere_material(m) {}

	__device__ virtual bool hit(const ray& r,double tmin,double tmax, intersection& isect);
};

#endif