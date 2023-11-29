#pragma once

#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
public:
	vec3 origin;
	vec3 direction;

	ray() {}
	ray(const vec3& o, const vec3& d) :direction(d), origin(o) {}

	vec3 at(double t) const { return origin + t * direction; }

};


#endif