#pragma once 

#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <vector>

#include "hitable.h"

class triangle : public hitable{
public:
	std::vector<vec3> vertices;
	material triangle_material;
	triangle() {}
	triangle(const vec3& v0, const vec3& v1, const vec3& v2,material m):triangle_material(m) {
		vertices.push_back(v0);
		vertices.push_back(v1);
		vertices.push_back(v2);
	}
	virtual bool hit(const ray& r, double tmin, double tmax, intersection& isect);
};

#endif