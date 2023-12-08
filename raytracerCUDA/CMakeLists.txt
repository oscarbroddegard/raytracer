add_executable(raytracerCUDA 
"main.cu"
 "headers/raytracer_cuda.h" 
 "headers/camera.h" "headers/color.h"
 "headers/hitable.h" "headers/material.h" "headers/ray.h" "headers/scene.h" "headers/sphere.h" 
 "headers/stb_image.h" "headers/stb_image_write.h" 
 "headers/triangle.h" "headers/vec3.h" 
 "sphere.cpp" "triangle.cpp")