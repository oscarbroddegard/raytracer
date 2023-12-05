#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"
#include "ray.h"

class camera {
public:
	camera(const vec3& o,const vec3& c,const vec3& u,const double& v,const double& a,int width,int height):
		eye(o),lookat(c),fov(v),aspectratio(a),imageheight(height),imagewidth(width) {
		
		forward = (lookat - eye).normalize();
		right = cross(forward, u).normalize();
		up = cross(right, forward);

		viewX = std::tan(0.5*fov*3.14159/180.0);
		viewY = std::tan(0.5 * fov / aspectratio * 3.14159 / 180.0);
	}

	ray getray(double x,double y) const{
		vec3 pdeltax = 2.0 / ((double)imagewidth) * viewX * right;
		vec3 pdeltay = -2.0 / ((double)imageheight) * viewX * up;
		vec3 view = forward - viewX * right + viewY * up;
		return ray(eye, view + x * pdeltax + y * pdeltay);
	}
private:
	vec3 eye,lookat,up,forward,right;
	double fov;
	double aspectratio;
	double viewX;
	double viewY;
	int imagewidth;
	int imageheight;
};

#endif