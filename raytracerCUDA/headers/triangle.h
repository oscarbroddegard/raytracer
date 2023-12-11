#pragma once 

#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <vector>

#include "hitable.h"

__host__  class triangle : public hitable{
public:
	vec3 vertices[3];
	material triangle_material;
	__host__  triangle() {}
	__host__  triangle(const vec3& v0, const vec3& v1, const vec3& v2,material m):triangle_material(m) {
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
	}
	virtual bool hit(const ray& r, float tmin, float tmax, intersection& isect);
};

#endif