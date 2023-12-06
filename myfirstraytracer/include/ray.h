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

inline ray getshadowray(const vec3& pos, const vec3& lightpos) {
	vec3 L = lightpos - pos;
	double Ldist = L.norm();
	ray lightray(pos, L.normalize());
	return lightray;
}

inline ray getreflectedray(const vec3& incident,const vec3& normal, const vec3& pos) {
	double ref = dot(incident, normal);
	return ray(pos, incident - 2.0 * normal * ref);
}

inline ray getrefractedray(const vec3& incident, const vec3& normal, const vec3& pos, const double& ratio) {
	double r = -dot(incident, normal);
	double c = 1.0 - ratio * ratio * (1 - r * r);
	vec3 dir = c > 0 ? ratio * incident + (ratio * r - sqrt(c)) * normal : getreflectedray(incident, normal, pos).direction;
	return ray(pos, dir);
}

#endif