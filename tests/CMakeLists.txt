# Test executables and CTest registration; enabled by the root BUILD_TESTING option.
find_program(LAB_UV_EXECUTABLE uv REQUIRED)
add_executable(lab_gpu_smoke integration/gpu_smoke.cpp)
target_include_directories(lab_gpu_smoke PRIVATE "${PROJECT_BINARY_DIR}/generated")
target_link_libraries(lab_gpu_smoke PRIVATE lab_compute mlx)
target_compile_options(lab_gpu_smoke PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(lab_gpu_smoke PROPERTIES
  BUILD_RPATH "${LAB_MLX_ROOT}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_test(NAME project.docs COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/checks/check_project.py" docs --root "${PROJECT_SOURCE_DIR}")
add_test(NAME project.build_boundary COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/checks/check_project.py" build --root "${PROJECT_SOURCE_DIR}"
  --build "${PROJECT_BINARY_DIR}" --binary "$<TARGET_FILE:pitfall-lab>"
  --gpu-binary "$<TARGET_FILE:lab_gpu_smoke>")
add_test(NAME dependencies.sdk COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/cmake/mlx/locate_sdk.py" --root "${LAB_MLX_ROOT}")
add_test(NAME dependencies.lock COMMAND "${LAB_UV_EXECUTABLE}" lock --offline --check
  --project "${PROJECT_SOURCE_DIR}")
set_tests_properties(dependencies.lock PROPERTIES
  ENVIRONMENT "UV_CACHE_DIR=${PROJECT_SOURCE_DIR}/.cache/uv;UV_NO_PROGRESS=1")
add_test(NAME checks.failure_cases COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/checks/test_checks.py")
add_test(NAME cli.contract COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/test_cli.py" --binary "$<TARGET_FILE:pitfall-lab>")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/test-work/gpu")
add_test(NAME gpu.smoke COMMAND "$<TARGET_FILE:lab_gpu_smoke>")
set_tests_properties(gpu.smoke PROPERTIES LABELS "gpu" RUN_SERIAL TRUE
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test-work/gpu")
set_tests_properties(project.docs project.build_boundary dependencies.sdk dependencies.lock
  checks.failure_cases cli.contract PROPERTIES LABELS "host")
get_property(lab_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${lab_tests} PROPERTIES TIMEOUT 60)
