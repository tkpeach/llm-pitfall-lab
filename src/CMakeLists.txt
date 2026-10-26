add_subdirectory(compute)
add_subdirectory(task)

add_executable(pitfall-lab main.cpp)
target_link_libraries(pitfall-lab PRIVATE lab_task)
target_include_directories(pitfall-lab PRIVATE "${PROJECT_BINARY_DIR}/generated")
target_compile_options(pitfall-lab PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(pitfall-lab PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
