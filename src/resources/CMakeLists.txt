add_library(lab_resources STATIC preflight.cpp writer.cpp)
target_link_libraries(lab_resources PRIVATE lab_storage)
target_include_directories(lab_resources
  PUBLIC "${PROJECT_SOURCE_DIR}/src"
  PRIVATE "${PROJECT_BINARY_DIR}/generated")
# 复用配置解析与报告序列化，资源判断和文件归属规则由本模块承担。
target_include_directories(lab_resources SYSTEM PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/tomlplusplus/include"
  "${PROJECT_SOURCE_DIR}/third_party/nlohmann/include")
target_compile_options(lab_resources PRIVATE -Wall -Wextra -Wpedantic)
