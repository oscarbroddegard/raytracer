#pragma once 

#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"

class material {
public:
	material() {}
	material(color c, double r,double t):diffuse_color(c),reflectivity(r),transparency(t) {}

	color diffuse_color;
	double reflectivity;
	double transparency;
};

#endif