// myfirstraytracer.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <conio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "sphere.h"
#include "hitable.h"
#include "camera.h"
#include "scene.h"
#include "material.h"
#include "triangle.h"