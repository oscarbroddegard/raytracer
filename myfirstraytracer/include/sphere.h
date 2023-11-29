#pragma once

#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "ray.h"

class sphere {
public:
	vec3 center;
	double radius;

	sphere() {}
	sphere(const vec3& c, const double& r) :center(c), radius(r) {}

	bool hit(const ray& r);
};

#endif