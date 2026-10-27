add_library(lab_runner STATIC control.cpp)
target_include_directories(lab_runner PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_compile_options(lab_runner PRIVATE -Wall -Wextra -Wpedantic)
