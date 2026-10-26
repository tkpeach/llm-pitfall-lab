# Test executables and CTest registration; enabled by the root BUILD_TESTING option.
find_program(LAB_UV_EXECUTABLE uv REQUIRED)
# 原生用例隔离校验/计分契约，CLI 用例验证文件到报告的完整链路；两者均不需要 GPU。
add_executable(lab_task_contract task/contract.cpp)
target_include_directories(lab_task_contract PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(lab_task_contract PRIVATE lab_task)
target_compile_options(lab_task_contract PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME task.contract COMMAND "$<TARGET_FILE:lab_task_contract>")
set_tests_properties(task.contract PROPERTIES LABELS "host")
add_test(NAME task.evaluate COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/test_evaluate.py" --binary "$<TARGET_FILE:pitfall-lab>"
  --fixtures "${PROJECT_SOURCE_DIR}/fixtures/task_contract.json")
set_tests_properties(task.evaluate PROPERTIES LABELS "host")
add_test(NAME task.dataset COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/checks/check_dataset.py"
  --root "${PROJECT_SOURCE_DIR}/fixtures/order-evaluation/v1" --binary "$<TARGET_FILE:pitfall-lab>")
add_test(NAME checks.dataset_failure_cases COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/checks/test_dataset.py"
  --root "${PROJECT_SOURCE_DIR}/fixtures/order-evaluation/v1")
set_tests_properties(task.dataset checks.dataset_failure_cases PROPERTIES LABELS "host")
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
