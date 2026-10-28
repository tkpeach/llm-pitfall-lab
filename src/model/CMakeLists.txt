add_library(lab_model STATIC qwen35.cpp)
target_include_directories(lab_model PUBLIC "${PROJECT_SOURCE_DIR}/src"
  PRIVATE "${PROJECT_SOURCE_DIR}/third_party/nlohmann/include")
target_link_libraries(lab_model PUBLIC mlx PRIVATE lab_storage lab_compute)
target_compile_options(lab_model PRIVATE -Wall -Wextra -Wpedantic)
