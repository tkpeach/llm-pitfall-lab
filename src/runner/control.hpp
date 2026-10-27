// 训练循环的信号暂停控制与安全更新边界保存协调。
#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <signal.h>
#include <string>

namespace lab::runner {

enum class BoundaryAction { continue_training, paused };

struct RunResult {
  std::uint64_t completed_steps;
  bool paused;
};

class PauseControl {
 public:
  PauseControl() = default;
  PauseControl(const PauseControl&) = delete;
  PauseControl& operator=(const PauseControl&) = delete;
  ~PauseControl();

  // Installs SIGINT/SIGTERM handlers. Call once at worker startup, before GPU work.
  std::expected<void, std::string> install();
  bool installed() const noexcept;
  bool requested() const noexcept;

  // Called only after a complete optimizer update. Save is invoked only when requested.
  std::expected<BoundaryAction, std::string> at_safe_boundary(
      const std::function<std::expected<void, std::string>()>& save);

 private:
  bool installed_ = false;
  struct sigaction old_sigint_{};
  struct sigaction old_sigterm_{};
};

using UpdateStep = std::function<std::expected<void, std::string>()>;
using SaveCheckpoint = std::function<std::expected<void, std::string>(std::uint64_t completed_steps)>;

// Checks pause at entry and after each complete update, then saves and returns.
std::expected<RunResult, std::string> run(
    PauseControl& control, std::uint64_t completed_steps, std::uint64_t total_steps,
    const UpdateStep& update, const SaveCheckpoint& save);

}  // namespace lab::runner
