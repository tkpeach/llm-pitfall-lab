# Test executables and CTest registration; enabled by the root BUILD_TESTING option.
find_program(LAB_UV_EXECUTABLE uv REQUIRED)
add_executable(lab_checkpoint_store checkpoint/store.cpp)
target_include_directories(lab_checkpoint_store PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(lab_checkpoint_store PRIVATE lab_checkpoint)
target_compile_options(lab_checkpoint_store PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME checkpoint.store COMMAND "$<TARGET_FILE:lab_checkpoint_store>")
set_tests_properties(checkpoint.store PROPERTIES LABELS "host")
add_executable(lab_resources_writer resources/writer.cpp)
target_include_directories(lab_resources_writer PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(lab_resources_writer PRIVATE lab_resources)
target_compile_options(lab_resources_writer PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME resources.writer COMMAND "$<TARGET_FILE:lab_resources_writer>")
set_tests_properties(resources.writer PROPERTIES LABELS "host")
add_test(NAME checkpoint.transactions COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/test_checkpoint_cli.py" --binary "$<TARGET_FILE:pitfall-lab>")
set_tests_properties(checkpoint.transactions PROPERTIES LABELS "host")
add_executable(lab_resources_preflight resources/preflight.cpp)
target_include_directories(lab_resources_preflight PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(lab_resources_preflight PRIVATE lab_resources)
target_compile_options(lab_resources_preflight PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME resources.preflight COMMAND "$<TARGET_FILE:lab_resources_preflight>")
set_tests_properties(resources.preflight PROPERTIES LABELS "host")
add_test(NAME resources.cli COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/test_resources_cli.py" --binary "$<TARGET_FILE:pitfall-lab>")
set_tests_properties(resources.cli PROPERTIES LABELS "host")
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
add_executable(lab_runner_control runner/control.cpp)
target_include_directories(lab_runner_control PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(lab_runner_control PRIVATE lab_runner)
target_compile_options(lab_runner_control PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME runner.control COMMAND "$<TARGET_FILE:lab_runner_control>")
set_tests_properties(runner.control PROPERTIES LABELS "host")
add_executable(lab_training_lifecycle integration/training_lifecycle.cpp)
target_include_directories(lab_training_lifecycle PRIVATE
  "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/third_party/nlohmann/include"
  "${PROJECT_BINARY_DIR}/generated")
target_link_libraries(lab_training_lifecycle PRIVATE lab_compute lab_checkpoint lab_runner mlx)
target_compile_options(lab_training_lifecycle PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(lab_training_lifecycle PROPERTIES BUILD_RPATH "${LAB_MLX_ROOT}/lib")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/test-work/training-lifecycle")
add_test(NAME training.lifecycle COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/test_training_lifecycle.py"
  --binary "$<TARGET_FILE:lab_training_lifecycle>")
set_tests_properties(training.lifecycle PROPERTIES LABELS "gpu" RUN_SERIAL TRUE
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test-work/training-lifecycle")
set_tests_properties(project.docs project.build_boundary dependencies.sdk dependencies.lock
  checks.failure_cases cli.contract PROPERTIES LABELS "host")
get_property(lab_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${lab_tests} PROPERTIES TIMEOUT 60)

add_test(NAME training.driver_cleanup COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/test_training_lifecycle_driver.py")
set_tests_properties(training.driver_cleanup PROPERTIES LABELS "host" TIMEOUT 15)
# Keep the driver alive long enough to perform its own timeout cleanup.
set_tests_properties(training.lifecycle PROPERTIES TIMEOUT 120)

add_executable(lab_gated_delta integration/gated_delta.cpp)
target_link_libraries(lab_gated_delta PRIVATE lab_compute mlx)
target_compile_options(lab_gated_delta PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(lab_gated_delta PROPERTIES BUILD_RPATH "${LAB_MLX_ROOT}/lib")
add_test(NAME compute.gated_delta COMMAND "$<TARGET_FILE:lab_gated_delta>")
set_tests_properties(compute.gated_delta PROPERTIES LABELS "gpu" RUN_SERIAL TRUE TIMEOUT 60)

add_executable(lab_qwen35_config model/config.cpp)
target_include_directories(lab_qwen35_config PRIVATE "${PROJECT_SOURCE_DIR}/third_party/nlohmann/include")
target_link_libraries(lab_qwen35_config PRIVATE lab_model)
set_target_properties(lab_qwen35_config PROPERTIES BUILD_RPATH "${LAB_MLX_ROOT}/lib")
add_test(NAME model.config COMMAND "$<TARGET_FILE:lab_qwen35_config>"
  "${PROJECT_SOURCE_DIR}/fixtures/qwen35-4b/config.json")
set_tests_properties(model.config PROPERTIES LABELS "host" TIMEOUT 30)

add_test(NAME checks.model_preparation COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/checks/test_model_preparation.py")
set_tests_properties(checks.model_preparation PROPERTIES LABELS "host" TIMEOUT 30)

# Large pinned artifacts are prepared explicitly, never downloaded by CTest.
set(LAB_QWEN35_ARTIFACT "" CACHE PATH "Verified Qwen3.5-4B language artifact for D4 reference acceptance")
if(LAB_QWEN35_ARTIFACT)
  if(NOT EXISTS "${LAB_QWEN35_ARTIFACT}/artifact.json")
    message(FATAL_ERROR "LAB_QWEN35_ARTIFACT requires a complete artifact.json; prepare the pinned model first")
  endif()
  add_test(NAME model.reference COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tests/model/test_qwen35_comparison.py"
    --root "${PROJECT_SOURCE_DIR}" --artifact "${LAB_QWEN35_ARTIFACT}"
    --binary "$<TARGET_FILE:lab_qwen35_probe>" --resource-binary "$<TARGET_FILE:pitfall-lab>"
    --report "${PROJECT_BINARY_DIR}/model-reference.json")
  set_tests_properties(model.reference PROPERTIES LABELS "gpu" RUN_SERIAL TRUE TIMEOUT 900)
endif()

add_test(NAME conversion.rules COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tests/checks/test_qwen35_conversion.py")
set_tests_properties(conversion.rules PROPERTIES LABELS "gpu" RUN_SERIAL TRUE TIMEOUT 60)
