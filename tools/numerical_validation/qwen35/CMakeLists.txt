# Observe the formal model implementation without linking or importing tests.
add_executable(lab_qwen35_probe probe.cpp)
target_include_directories(lab_qwen35_probe PRIVATE "${PROJECT_SOURCE_DIR}/third_party/nlohmann/include")
target_link_libraries(lab_qwen35_probe PRIVATE lab_model lab_compute lab_resources)
target_compile_options(lab_qwen35_probe PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(lab_qwen35_probe PROPERTIES BUILD_RPATH "${LAB_MLX_ROOT}/lib")
