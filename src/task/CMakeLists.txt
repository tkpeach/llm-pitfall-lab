add_library(lab_task STATIC evaluator.cpp)
target_include_directories(lab_task
  PUBLIC "${PROJECT_SOURCE_DIR}/src"
  PRIVATE "${PROJECT_BINARY_DIR}/generated")
# JSON 库只服务于任务模块内部；上层通过任务类型和报告接口使用评估，不依赖解析库类型。
target_include_directories(lab_task SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/nlohmann/include")
target_compile_options(lab_task PRIVATE -Wall -Wextra -Wpedantic)
