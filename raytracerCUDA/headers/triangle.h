#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <vector>

#include "hitable.h"

class triangle : public hitable {
public:
	vec3** vertices;
	material* triangle_material;
	__device__ triangle() {}
	__device__ triangle(vec3* v0, vec3* v1, vec3* v2, material* m) :triangle_material(m) {
		*vertices = v0;
		*(vertices + 1) = v1;
		*(vertices + 2) = v2;
	}
	__device__ virtual bool hit(const ray& r, double tmin, double tmax, intersection& isect) const;
};


__device__ bool triangle::hit(const ray& r, double tmin, double tmax, intersection& isect) const{
	vec3 e1 = *vertices[1] - *vertices[0];
	vec3 e2 = *vertices[2] - *vertices[0];
	vec3 N = cross(e1, e2);
	double t = -(dot(N, r.getorigin() - *vertices[0])) / (dot(N, r.getdirection()));
	vec3 Q = r.getorigin() + t * r.getdirection();
	vec3 u = Q - *vertices[0];
	vec3 V = cross(e1, u);
	vec3 W = cross(u, e2);
	double v = dot(V, N);
	double w = dot(W, N);
	double inside = (V.norm() + W.norm()) / N.norm();

	if (v > 0 && w > 0 && inside < 1 && t < tmax && t > tmin) {
		isect.hit_t = t;
		isect.hit_normal = dot(-r.direction, N) > 0.0 ? N.normalize() : -N.normalize();
		isect.hit_position = Q;
		isect.hit_material = triangle_material;
		return true;
	}
	return false;
}

#endif