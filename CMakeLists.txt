cmake_minimum_required(VERSION 3.16)
project(fire_scene_enhancer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_library(fse STATIC
  src/common.cpp
  src/mask.cpp
  src/dehaze.cpp
  src/clahe.cpp
  src/enhancer.cpp
  src/metrics.cpp
  src/frame_reader.cpp
)
target_include_directories(fse PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fse PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX Threads::Threads)

if(MSVC)
  target_compile_options(fse PRIVATE /W3)
else()
  target_compile_options(fse PRIVATE -Wall -Wextra)
endif()

add_executable(fire_enhance src/main.cpp)
target_link_libraries(fire_enhance PRIVATE fse)
