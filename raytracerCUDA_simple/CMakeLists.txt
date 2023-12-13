

set(CUDA_SOURCE_FILES_SIMPLE
    main.cu
)
cmake_policy(SET CMP0104 NEW)
set(CUDA_ARCHITECTURES "sm_86" CACHE STRING "CUDA architectures to build for")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=32 --ptxas-options=-v")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(raytracer_cuda_simple 
 ${CUDA_SOURCE_FILES_SIMPLE}
 "simpleRT.h"
 "stb_image.h"
 "stb_image_write.h"
 "vec3.h"
 "color.h"
 "ray.h"
 "camera.h")