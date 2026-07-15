cmake_minimum_required(VERSION 3.10)
if(WIN32)
    set(CMAKE_TOOLCHAIN_FILE "C:/PROJ/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain")
endif()
project(avm)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type and optimization options
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Print OpenCV information
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Include OpenCV header files
include_directories(${OpenCV_INCLUDE_DIRS})

# Add header file directory
include_directories(include)

# Collect source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create executable file
add_executable(avm ${SOURCES})

# Link OpenCV libraries
target_link_libraries(avm ${OpenCV_LIBS})

# Set output directory
set_target_properties(avm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
