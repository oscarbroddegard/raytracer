#include "headers/sphere.h"

bool sphere::hit(const ray& r,double tmin,double tmax,intersection& isect) {
	vec3 OC = r.origin - center;
	double a = dot(r.direction, r.direction);
	double b = dot(OC, r.direction);
	double c = dot(OC,OC)-radius*radius;

	double disc = b * b - a * c;
	if (disc > 0) {
		double temp = (-b-sqrt(disc)) / a;
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