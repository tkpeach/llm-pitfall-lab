// 信号处理器只设置 sig_atomic_t 标记；检查点保存留在普通训练控制流中执行。
#include "runner/control.hpp"

#include <cerrno>
#include <cstring>
#include <utility>

namespace lab::runner {
namespace {
volatile sig_atomic_t pause_requested = 0;
volatile sig_atomic_t handler_installed = 0;

void request_pause(int) { pause_requested = 1; }

std::string system_error(const char* operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}
}  // namespace

PauseControl::~PauseControl() {
  if (!installed_) return;
  ::sigaction(SIGTERM, &old_sigterm_, nullptr);
  ::sigaction(SIGINT, &old_sigint_, nullptr);
  handler_installed = 0;
  pause_requested = 0;
}

std::expected<void, std::string> PauseControl::install() {
  if (installed_) return std::unexpected("pause control already installed");
  if (handler_installed) return std::unexpected("pause control already active in this process");

  struct sigaction action {};
  action.sa_handler = request_pause;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;

  pause_requested = 0;
  if (::sigaction(SIGINT, &action, &old_sigint_) != 0)
    return std::unexpected(system_error("install SIGINT handler"));
  if (::sigaction(SIGTERM, &action, &old_sigterm_) != 0) {
    const auto error = system_error("install SIGTERM handler");
    ::sigaction(SIGINT, &old_sigint_, nullptr);
    return std::unexpected(error);
  }
  installed_ = true;
  handler_installed = 1;
  return {};
}

bool PauseControl::requested() const noexcept { return pause_requested != 0; }
bool PauseControl::installed() const noexcept { return installed_; }

std::expected<BoundaryAction, std::string> PauseControl::at_safe_boundary(
    const std::function<std::expected<void, std::string>()>& save) {
  if (!installed_) return std::unexpected("pause control is not installed");
  if (!requested()) return BoundaryAction::continue_training;
  if (!save) return std::unexpected("pause checkpoint callback is empty");
  const auto saved = save();
  if (!saved) return std::unexpected("pause checkpoint failed: " + saved.error());
  return BoundaryAction::paused;
}

std::expected<RunResult, std::string> run(
    PauseControl& control, std::uint64_t completed_steps, std::uint64_t total_steps,
    const UpdateStep& update, const SaveCheckpoint& save) {
  if (!control.installed())
    return std::unexpected("pause control must be installed before training");
  if (completed_steps > total_steps)
    return std::unexpected("completed steps exceed total steps");
  if (!update) return std::unexpected("training update callback is empty");

  // The restored/initial state is already a safe boundary. Check it before
  // starting GPU work, as well as after every complete optimizer update.
  for (;;) {
    auto action = control.at_safe_boundary([&]() -> std::expected<void, std::string> {
      if (!save) return std::unexpected("pause checkpoint callback is empty");
      return save(completed_steps);
    });
    if (!action) return std::unexpected(action.error());
    if (*action == BoundaryAction::paused) return RunResult{completed_steps, true};
    if (completed_steps == total_steps) return RunResult{completed_steps, false};
    const auto updated = update();
    if (!updated) return std::unexpected("training update failed: " + updated.error());
    ++completed_steps;
  }
}

}  // namespace lab::runner
