# CMakeList.txt : CMake project for myfirstraytracer, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (myfirstraytracer "myfirstraytracer.cpp"
"include/myfirstraytracer.h" 
"include/color.h" "include/vec3.h" "include/ray.h" "include/hitable.h"
"include/sphere.h" "sphere.cpp" 
"include/triangle.h" "triangle.cpp"
"include/stb_image.h" "include/stb_image_write.h" "include/camera.h" "include/material.h" "include/scene.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET myfirstraytracer PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
