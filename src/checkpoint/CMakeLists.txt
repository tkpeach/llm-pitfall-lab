add_library(lab_checkpoint STATIC store.cpp)
target_include_directories(lab_checkpoint PUBLIC "${PROJECT_SOURCE_DIR}/src"
  PRIVATE "${PROJECT_BINARY_DIR}/generated")
target_include_directories(lab_checkpoint SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/nlohmann/include")
target_link_libraries(lab_checkpoint PUBLIC lab_resources PRIVATE lab_storage)
target_compile_options(lab_checkpoint PRIVATE -Wall -Wextra -Wpedantic)
