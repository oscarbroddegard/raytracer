// myfirstraytracer.h : Include file for standard system include files,
// or project specific include files.

#pragma once
//cxx standards
#include <conio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

//cuda essentials
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//project specific
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "headers/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "headers/stb_image.h"

#include "headers/vec3.h"
#include "headers/color.h"
#include "headers/ray.h"
#include "headers/sphere.h"
#include "headers/hitable.h"
#include "headers/camera.h"
#include "headers/scene.h"
#include "headers/material.h"
#include "headers/triangle.h"