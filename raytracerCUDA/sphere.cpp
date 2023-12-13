#include "headers/sphere.h"

bool sphere::hit(const ray& r,float tmin,float tmax,intersection& isect) {
	vec3 OC = r.getorigin() - center;
	float a = dot(r.getdirection(), r.getdirection());
	float b = dot(OC, r.getdirection());
	float c = dot(OC,OC)-radius*radius;

	float disc = b * b - a * c;
	if (disc > 0) {
		float temp = (-b-sqrt(disc)) / a;
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

