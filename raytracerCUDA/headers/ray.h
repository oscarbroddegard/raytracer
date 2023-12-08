#pragma once

#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
public:
	vec3 origin;
	vec3 direction;

	__device__ ray() {}
	__device__ ray(const vec3& o, const vec3& d) :direction(d), origin(o) {}

	__device__ vec3 at(float t) const { return origin + t * direction; }

};

__device__ inline ray getshadowray(const vec3& pos, const vec3& lightpos) {
	vec3 L = lightpos - pos;
	double Ldist = L.norm();
	ray lightray(pos, L.normalize());
	return lightray;
}

__device__ inline ray getreflectedray(const vec3& incident,const vec3& normal, const vec3& pos) {
	float ref = dot(incident, normal);
	return ray(pos, incident - 2.0f * normal * ref);
}

__device__ inline ray getrefractedray(const vec3& incident, const vec3& normal, const vec3& pos, const float& ratio) {
	float r = -dot(incident, normal);
	float c = 1.0f - ratio * ratio * (1.0f - r * r);
	vec3 dir = c > 0.0f ? ratio * incident + (ratio * r - sqrt(c)) * normal : getreflectedray(incident, normal, pos).direction;
	return ray(pos, dir);
}

#endif