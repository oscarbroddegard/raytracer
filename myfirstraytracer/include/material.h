#pragma once 

#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"

class material {
public:
	material() {}
	material(color c, double r,double t,double i):diffuse_color(c),reflectivity(r),transparency(t),refractive_index(i) {}

	color diffuse_color;
	double reflectivity;
	double transparency;
	double refractive_index;
};

#endif