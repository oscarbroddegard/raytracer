#pragma once 

#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"

class material {
public:
	__host__ __device__ material() {}
	__host__ __device__ material(vec3 c, double r,double t,double i):diffuse_color(c),reflectivity(r),transparency(t),refractive_index(i) {}

	vec3 diffuse_color;
	double reflectivity;
	double transparency;
	double refractive_index;
};

#endif