cmake_minimum_required(VERSION 3.20)
project(cuda-learn LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

include_directories(${CMAKE_SOURCE_DIR}/common)
include_directories(${CMAKE_SOURCE_DIR}/stb)

find_package(SDL2 QUIET)
if(SDL2_FOUND)
  message(STATUS "SDL2 found — interactive visualization enabled")
  add_compile_definitions(USE_SDL2)
  include_directories(${SDL2_INCLUDE_DIRS})
endif()

add_subdirectory(stage0_warmup)
add_subdirectory(stage1_pde)
add_subdirectory(stage2_opticalflow)
add_subdirectory(stage3_interpolation)
