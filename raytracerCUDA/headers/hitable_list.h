#ifndef HITABLELIST_H
#define HITABLELIST_H


#include "sphere.h"

class sphere_list{
public:
	__host__ __device__ sphere_list() {}

	__host__ __device__ sphere_list(sphere* l, int n) { list = l; n_hitables = n; }
	__device__ bool hit(const ray& r, float tmin, float tmax, intersection& isect);
	sphere* list;
	int n_hitables;
};


__device__ bool sphere_list::hit(const ray& r, float tmin, float tmax, intersection& isect) {
	bool hit = false;
	intersection temp_isect;
	double closest_t = tmax;
	for (int k = 0; k < n_hitables; k++) {
		if (sphere_hit(r, &list[k], tmin, closest_t, temp_isect)) {
			hit = true;
			closest_t = temp_isect.hit_t;
			isect = temp_isect;
		}
	}
	return hit;
}

#endif