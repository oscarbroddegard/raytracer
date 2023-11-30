#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"
#include "ray.h"

class camera {
public:
	camera(vec3 c,double focal_length, double image_width,double image_height, double viewport_height):origin(c) {
		auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);
		vec3 viewport_u(viewport_width, 0, 0);
		vec3 viewport_v(0, -viewport_height, 0);

		pdeltax = viewport_u / image_width;
		pdeltay = viewport_v / image_height;

		vec3 view_upperleft = origin - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
		pixel00location = view_upperleft + 0.5 * (pdeltax + pdeltay);
	}

	ray getray(int i,int j) const{
		vec3 pixel_center = pixel00location + j * pdeltay + i * pdeltax;

		return ray(origin, pixel_center - origin);
	}
private:
	vec3 origin;
	vec3 pdeltax;
	vec3 pdeltay;
	vec3 pixel00location;
};

#endif