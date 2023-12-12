


set(CUDA_SOURCE_FILES
    main.cu
)
cmake_policy(SET CMP0104 NEW)
set(CUDA_ARCHITECTURES "sm_86" CACHE STRING "CUDA architectures to build for")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=32 --ptxas-options=-v")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(raytracer_cuda 
 ${CUDA_SOURCE_FILES}
 "raytracer_cuda.h" 
    "headers/camera.h" "headers/color.h"
     "headers/hitable.h" "headers/hitable_list.h" "headers/material.h" "headers/ray.h" "headers/scene.h" "headers/sphere.h" 
     "headers/stb_image.h" "headers/stb_image_write.h" 
     "headers/triangle.h" "headers/vec3.h" 
     "sphere.cpp" #"triangle.cpp"
)