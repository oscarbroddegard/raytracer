# Find Vulkan package
find_package(Vulkan REQUIRED)

# Find GLFW library
find_package(glfw3 REQUIRED)

# Find GLM library
find_package(glm REQUIRED)

# Include directories
include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})

add_executable(vulkan_raytracer
"main.cpp"
)

# Link libraries
target_link_libraries(vulkan_raytracer ${Vulkan_LIBRARIES})
target_link_libraries(vulkan_raytracer glfw)
# No need to explicitly link against GLM, as it is a header-only library