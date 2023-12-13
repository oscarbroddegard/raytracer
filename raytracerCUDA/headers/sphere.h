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

	__host__ __device__ sphere() {}
	__host__ __device__ sphere(const vec3& c, const float& r, material m) :center(c), radius(r) { sphere_material = m; }

	virtual bool hit(const ray& r,float tmin,float tmax, intersection& isect);
};

__device__ inline bool sphere_hit(const ray& r, const sphere& target, float tmin, float tmax, intersection& isect) {
	vec3 OC = r.getorigin() - target.center;
	float a = dot(r.getdirection(), r.getdirection());
	float b = dot(OC, r.getdirection());
	float c = dot(OC, OC) - target.radius * target.radius;

	float disc = b * b - a * c;
	if (disc > 0) {
		float temp = (-b - sqrt(disc)) / a;
		if (temp < tmax && temp > tmin) {
			isect.hit_t = temp;
			isect.hit_position = r.at(isect.hit_t);
			isect.hit_normal = (isect.hit_position - target.center) / target.radius;
			isect.hit_material = target.sphere_material;
			return true;
		}
		temp = (-b + sqrt(disc)) / a;
		if (temp < tmax && temp > tmin) {
			isect.hit_t = temp;
			isect.hit_position = r.at(isect.hit_t);
			isect.hit_normal = ((isect.hit_position - target.center) / target.radius).normalize();
			isect.hit_material = target.sphere_material;
			return true;
		}
	}

	return false;
}

#endif