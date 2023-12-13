#ifndef HITABLELIST_H
#define HITABLELIST_H


#include "sphere.h"


class hitable_list : public hitable {
public:
	__device__ hitable_list() {}
	__device__ hitable_list(hitable** l, int n) { list = l; n_hitables = n; }
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, intersection& isect) const;
	hitable** list;
	int n_hitables;
};

__device__ bool hitable_list::hit(const ray& r, float tmin, float tmax, intersection& isect) const{
	bool hit = false;
	intersection temp_isect;
	double closest_t = tmax;
	for (int k = 0; k < n_hitables; k++) {
		if (list[k]->hit(r, tmin, closest_t, temp_isect)) {
			hit = true;
			closest_t = temp_isect.hit_t;
			isect = temp_isect;
		}
	}
	return hit;
}

#endif