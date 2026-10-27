// 验证正式 runner 在完整更新后响应信号、保存成功才暂停，并传播保存错误。
#include "runner/control.hpp"

#include <csignal>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main() {
  try {
    {
      lab::runner::PauseControl control;
      const auto installed = control.install();
      check(installed.has_value(), "pause control installation failed");
      int updates = 0;
      int saves = 0;
      std::uint64_t saved_step = 0;
      const auto result = lab::runner::run(control, 0, 5, [&]() -> std::expected<void, std::string> {
        ++updates;
        if (updates == 2) std::raise(SIGINT);
        return {};
      }, [&](std::uint64_t step) -> std::expected<void, std::string> {
        ++saves;
        saved_step = step;
        return {};
      });
      check(result.has_value(), "paused run returned an error");
      check(result->paused && result->completed_steps == 2, "pause was not reported at step 2");
      check(updates == 2 && saves == 1 && saved_step == 2, "save did not follow the completed update");
    }

    for (const auto initial_step : {0ULL, 3ULL, 8ULL}) {
      lab::runner::PauseControl control;
      check(control.install().has_value(), "pending pause control installation failed");
      std::raise(SIGTERM);
      int updates = 0;
      int saves = 0;
      std::uint64_t saved_step = 99;
      const auto result = lab::runner::run(control, initial_step, 8,
          [&]() -> std::expected<void, std::string> { ++updates; return {}; },
          [&](std::uint64_t step) -> std::expected<void, std::string> {
            ++saves;
            saved_step = step;
            return {};
          });
      check(result.has_value() && result->paused && result->completed_steps == initial_step,
            "pending pause changed the committed progress");
      check(updates == 0 && saves == 1 && saved_step == initial_step,
            "pending pause executed an update or saved the wrong progress");
    }

    {
      lab::runner::PauseControl control;
      check(control.install().has_value(), "pause control reinstall failed");
      std::raise(SIGTERM);
      int saves = 0;
      const auto result = lab::runner::run(control, 0, 3,
          []() -> std::expected<void, std::string> { return {}; },
          [&](std::uint64_t) -> std::expected<void, std::string> {
            ++saves;
            return std::unexpected("simulated storage failure");
          });
      check(!result.has_value(), "checkpoint failure was reported as a successful pause");
      check(result.error().find("simulated storage failure") != std::string::npos,
            "checkpoint failure reason was lost");
      check(saves == 1, "checkpoint callback was not called exactly once");
    }

    {
      lab::runner::PauseControl control;
      check(control.install().has_value(), "pause control install after failure failed");
      int updates = 0;
      int saves = 0;
      const auto result = lab::runner::run(control, 0, 4,
          [&]() -> std::expected<void, std::string> { ++updates; return {}; },
          [&](std::uint64_t) -> std::expected<void, std::string> { ++saves; return {}; });
      check(result.has_value() && !result->paused && result->completed_steps == 4,
            "uninterrupted run did not complete");
      check(updates == 4 && saves == 0, "uninterrupted run used the pause checkpoint callback");
    }

    std::cout << "PASS: runner pause boundary, checkpoint failure and uninterrupted control\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "runner control test failed: " << error.what() << '\n';
    return 1;
  }
}
