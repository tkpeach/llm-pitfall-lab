// 定义磁盘策略、卷空间快照与峰值决策接口；轻量实现不递归统计项目文件占用。
// 项目使用量未知时明确返回空值，不代表已预留额度或具备并发写入保护。

#pragma once
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lab::resources {
struct Policy {
  std::uint64_t project_limit_bytes;
  std::uint64_t warning_bytes;
  std::uint64_t min_free_bytes;
  std::vector<std::filesystem::path> extra_owned_paths;
};
struct Snapshot {
  // Current lightweight inspector leaves project totals unknown instead of walking every file.
  std::optional<std::uint64_t> allocated_bytes;
  std::uint64_t available_bytes = 0;
};
struct Decision {
  bool allowed = false;
  bool warning = false;
  std::uint64_t max_growth_bytes = 0;
  std::uint64_t requested_peak_bytes = 0;
  std::string reason;
};

// 配置无效即拒绝，不用默认阈值掩盖拼写、类型或缺失项错误。
std::expected<Policy, std::string> parse_policy(std::string_view text);
std::expected<Policy, std::string> load_policy(const std::filesystem::path& file);
// 校验项目根、显式外部目录和卷空间；当前实现不递归统计项目文件。
std::expected<Snapshot, std::string> inspect(const std::filesystem::path& root, const Policy& policy);
// peak 是预计新增占用的临时峰值。项目用量未知时只按卷安全线判断，不预留额度。
std::expected<Decision, std::string> assess(const Policy& policy, const Snapshot& snapshot,
                                         std::uint64_t peak);
std::string report_json(const Policy& policy, const Snapshot& snapshot, const Decision& decision);
}  // namespace lab::resources
