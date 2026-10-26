add_library(lab_storage STATIC files.cpp)
target_include_directories(lab_storage PUBLIC "${PROJECT_SOURCE_DIR}/src"
  PRIVATE "${PROJECT_BINARY_DIR}/generated")
target_compile_options(lab_storage PRIVATE -Wall -Wextra -Wpedantic)
