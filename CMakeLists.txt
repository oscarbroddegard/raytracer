# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)
project("myfirstraytracer")
# Enable CUDA
find_package(CUDA QUIET)

if(CUDA_FOUND)


	# Enable Hot Reload for MSVC compilers if supported.
	if (POLICY CMP0141)
	  cmake_policy(SET CMP0141 NEW)
	  set(CUDA_ARCHITECTURES "sm_86" CACHE STRING "CUDA architectures to build for")
	  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
	endif()


	project ("myfirstraytracer" LANGUAGES CXX CUDA)

	set(CUDA_ARCHITECTURES "sm_86" CACHE STRING "CUDA architectures to build for")
	set(CMAKE_CUDA_ARCHITECTURES 86)

	# Set CUDA standard (optional)
	#set(CMAKE_CUDA_STANDARD 11)
	#set(CMAKE_CUDA_STANDARD_REQUIRED ON)


	# Include sub-projects.
	add_subdirectory ("myfirstraytracer")
	add_subdirectory("raytracerCUDA")
	add_subdirectory("raytracerCUDA_simple")
	add_subdirectory("raytracinginoneweekend")
else()
	project("myfirstraytracer" LANGUAGES CXX)
	add_subdirectory ("myfirstraytracer")
endif()