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
	material *sphere_material;

	__host__ __device__ sphere() {}
	__host__ __device__ sphere(const vec3& c, const float& r, material* m) :center(c), radius(r),sphere_material(m) {}

	__device__ virtual bool hit(const ray& r,float tmin,float tmax, intersection& isect) const;
};

__device__ bool sphere::hit(const ray& r, float tmin, float tmax, intersection& isect) const{
	vec3 OC = r.getorigin() - center;
	float a = dot(r.getdirection(), r.getdirection());
	float b = dot(OC, r.getdirection());
	float c = dot(OC, OC) - radius * radius;

	float disc = b * b - a * c;
	if (disc > 0) {
		float temp = (-b - sqrt(disc)) / a;
		if (temp < tmax && temp > tmin) {
			isect.hit_t = temp;
			isect.hit_position = r.at(isect.hit_t);
			isect.hit_normal = (isect.hit_position - center) / radius;
			isect.hit_material = sphere_material;
			return true;
		}
		temp = (-b + sqrt(disc)) / a;
		if (temp < tmax && temp > tmin) {
			isect.hit_t = temp;
			isect.hit_position = r.at(isect.hit_t);
			isect.hit_normal = ((isect.hit_position - center) / radius).normalize();
			isect.hit_material = sphere_material;
			return true;
		}
	}

	return false;
}

#endif