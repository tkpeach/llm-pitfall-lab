add_subdirectory(storage)
add_subdirectory(compute)
add_subdirectory(task)
add_subdirectory(resources)
add_subdirectory(checkpoint)
add_subdirectory(runner)

add_executable(pitfall-lab main.cpp)
target_link_libraries(pitfall-lab PRIVATE lab_task lab_checkpoint)
target_include_directories(pitfall-lab PRIVATE "${PROJECT_BINARY_DIR}/generated")
target_compile_options(pitfall-lab PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(pitfall-lab PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
