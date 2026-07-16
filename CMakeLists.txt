cmake_minimum_required(VERSION 3.10)
if(WIN32)
    set(CMAKE_TOOLCHAIN_FILE "C:/PROJ/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain")
endif()
project(avm)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型和优化选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
    # /Od: 禁用优化 (最低优化)
    # /Zi: 生成完整的调试信息
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
else()
    # -O0: 禁用优化 (最低优化)
    # -g: 生成调试信息
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 打印 OpenCV 信息
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加头文件目录
include_directories(src)

# 收集源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 创建可执行文件
add_executable(avm ${SOURCES})

# 链接 OpenCV 库
target_link_libraries(avm ${OpenCV_LIBS})

# 设置输出目录
set_target_properties(avm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
