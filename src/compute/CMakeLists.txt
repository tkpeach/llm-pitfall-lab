add_library(lab_compute STATIC mlx_runtime.cpp gated_delta.cpp)
target_include_directories(lab_compute
  PUBLIC "${PROJECT_SOURCE_DIR}/src"
  PRIVATE "${PROJECT_BINARY_DIR}/generated")
target_link_libraries(lab_compute PRIVATE mlx)
target_compile_options(lab_compute PRIVATE -Wall -Wextra -Wpedantic)
