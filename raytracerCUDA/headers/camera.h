#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"
#include "ray.h"

class camera {
public:
	__host__ __device__ camera(const vec3& o,const vec3& c,const vec3& u,const float& v,const float& a,int width,int height):
		eye(o),lookat(c),fov(v),aspectratio(a),imageheight(height),imagewidth(width) {
		
		forward = (lookat - eye).normalize();
		right = cross(forward, u).normalize();
		up = cross(right, forward);

		viewX = std::tan(0.5*fov*3.14159/180.0);
		viewY = std::tan(0.5 * fov / aspectratio * 3.14159 / 180.0);
	}

	__host__ __device__ ray getray(float x,float y) const{
		vec3 pdeltax = 2.0 / ((float)imagewidth) * viewX * right;
		vec3 pdeltay = -2.0 / ((float)imageheight) * viewX * up;
		vec3 view = forward - viewX * right + viewY * up;
		return ray(eye, (view + x * pdeltax + y * pdeltay).normalize());
	}
private:
	vec3 eye,lookat,up,forward,right;
	float fov, aspectratio, viewX, viewY;
	int imagewidth, imageheight;
};

#endif