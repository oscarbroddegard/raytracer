#pragma once

#include "headers/triangle.h"

 bool triangle::hit(const ray& r, float tmin, float tmax, intersection& isect) {
	vec3 e1 = vertices[1] - vertices[0];
	vec3 e2 = vertices[2] - vertices[0];
	vec3 N = cross(e1, e2);
	float t = -(dot(N, r.origin - vertices[0])) / (dot(N, r.direction));
	vec3 Q = r.origin + t * r.direction;
	vec3 u = Q - vertices[0];
	vec3 V = cross(e1, u);
	vec3 W = cross(u, e2);
	float v = dot(V, N);
	float w = dot(W, N);
	float inside = (V.norm() + W.norm()) / N.norm();

	if (v > 0 && w > 0 && inside < 1 && t < tmax && t > tmin) {
		isect.hit_t = t;
		isect.hit_normal = dot(-r.direction,N) > 0.0 ? N.normalize() : -N.normalize();
		isect.hit_position = Q;
		isect.hit_material = triangle_material;
		return true;
	}
	return false;
}