#pragma once

#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "hitable.h"
#include "sphere.h"

class scene {
public:
	scene() {
		
	}

	void add_hitable(std::shared_ptr<hitable> h) {}
	void add_light(vec3 lpos) { light_sources.push_back(lpos); }

	std::vector<std::shared_ptr<hitable>> hitables;
	std::vector<vec3> light_sources;
};

#endif