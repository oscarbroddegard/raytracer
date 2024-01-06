# Specify the absolute paths to GLFW and GLM libraries
set(GLFW_DIR "C:/Users/oscar/Documents/Visual Studio 2022/Libraries/glfw-3.3.9.bin.WIN64")
set(GLM_DIR "C:/Users/oscar/Documents/Visual Studio 2022/Libraries/glm-0.9.9.8")

# Find Vulkan package
find_package(Vulkan REQUIRED)

# Include directories
include_directories(${Vulkan_INCLUDE_DIRS})

add_executable(vulkan_raytracer
"main.cpp"
)

# Link libraries
target_link_libraries(vulkan_raytracer ${Vulkan_LIBRARIES})
target_link_libraries(vulkan_raytracer glfw)
# No need to explicitly link against GLM, as it is a header-only library