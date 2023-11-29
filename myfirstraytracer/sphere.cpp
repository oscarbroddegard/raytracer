#include "include/sphere.h"


bool sphere::hit(const ray& r) {
	vec3 OC = r.origin - center;
	double a = dot(r.direction, r.direction);
	double b = 2.0 * dot(OC, r.direction);
	double c = dot(OC,OC)-radius*radius;

	double disc = b * b - 4 * a * c;

	return disc > 0;
}