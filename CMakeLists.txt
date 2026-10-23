cmake_minimum_required(VERSION 3.25)
project(llm_pitfall_lab VERSION 0.1.0 LANGUAGES CXX)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "Use an independent build directory, for example runs/d0/build")
endif()
if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  message(FATAL_ERROR "This project targets Apple Silicon macOS")
endif()
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Python supplies the SDK and auxiliary checks; it is not linked into the worker.
find_package(Python3 3.12...<3.13 REQUIRED COMPONENTS Interpreter)
include(cmake/mlx/MLXSDK.cmake)
lab_find_mlx()

configure_file(cmake/build_info.hpp.in generated/lab_build_info.hpp @ONLY)
add_subdirectory(src)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
