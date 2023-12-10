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


// helper functions

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void bindSceneBuffer(hitable* world,int n_hitables, hitable** buffer_ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int n = 0; n < n_hitables; n++) {
            *(buffer_ptr + n) = &world[n];
        }
    }
}

__global__ void freeBuffers(hitable** scenebuffer, int sb_size,camera** cam_ptr) {
    for (int n = 0; n < sb_size; n++) {
        delete* (scenebuffer + n);
    }
    delete* cam_ptr;
}