// 加载 TOML 策略，轻量读取卷可用空间并判断新增峰值；不遍历项目文件。
// 项目占用保持未知，不修改产物、创建额度预留或实现检查点提交。

#include "resources/preflight.hpp"
#include <nlohmann/json.hpp>
#include <toml++/toml.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <sys/stat.h>

namespace lab::resources {
namespace {
namespace fs = std::filesystem;
void validate_policy(const Policy& policy) {
  if (policy.project_limit_bytes == 0 || policy.warning_bytes == 0 ||
      policy.warning_bytes > policy.project_limit_bytes || policy.min_free_bytes == 0)
    throw std::runtime_error("require 0 < warning <= project limit and a positive free-space floor");
  for (const auto& path : policy.extra_owned_paths)
    if (!path.is_absolute()) throw std::runtime_error("extra_owned_paths must contain absolute paths");
}
}  // namespace

std::expected<Policy, std::string> parse_policy(std::string_view text) {
  try {
    const auto table = toml::parse(text);
    const std::set<std::string> names{
        "project_limit_bytes", "warning_bytes", "min_free_bytes", "extra_owned_paths"};
    if (table.size() != names.size()) throw std::runtime_error("resource policy requires exactly four keys");
    for (const auto& [key, value] : table) {
      (void)value;
      if (!names.contains(std::string(key.str()))) throw std::runtime_error("unknown resource policy key");
    }
    auto bytes = [&](std::string_view key) {
      const auto node = table[key];
      if (!node.is_integer() || node.as_integer()->get() <= 0)
        throw std::runtime_error(std::string(key) + " must be a positive TOML integer");
      return static_cast<std::uint64_t>(node.as_integer()->get());
    };
    Policy policy{bytes("project_limit_bytes"), bytes("warning_bytes"), bytes("min_free_bytes"), {}};
    const auto* extras = table["extra_owned_paths"].as_array();
    if (!extras) throw std::runtime_error("extra_owned_paths must be an array of absolute directory strings");
    for (const auto& value : *extras) {
      if (!value.is_string()) throw std::runtime_error("extra_owned_paths entries must be strings");
      policy.extra_owned_paths.emplace_back(value.as_string()->get());
    }
    validate_policy(policy);
    return policy;
  } catch (const std::exception& error) {
    return std::unexpected(error.what());
  }
}

std::expected<Policy, std::string> load_policy(const fs::path& file) {
  try {
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open resource policy: " + file.string());
    const auto size = input.tellg();
    if (size < 0 || size > 64 * 1024) throw std::runtime_error("resource policy exceeds 64 KiB or is unreadable");
    std::string contents(static_cast<std::size_t>(size), '\0');
    input.seekg(0);
    if (!input.read(contents.data(), static_cast<std::streamsize>(size)))
      throw std::runtime_error("cannot read resource policy");
    return parse_policy(contents);
  } catch (const std::exception& error) {
    return std::unexpected(error.what());
  }
}

std::expected<Snapshot, std::string> inspect(const fs::path& root, const Policy& policy) {
  try {
    validate_policy(policy);
    const auto project = fs::canonical(root);
    struct stat info{};
    if (::lstat(project.c_str(), &info) != 0)
      throw std::runtime_error("cannot stat project root: " + std::string(std::strerror(errno)));
    if (!S_ISDIR(info.st_mode)) throw std::runtime_error("project root must be a directory");
    Snapshot snapshot{std::nullopt, 0};
    for (const auto& path : policy.extra_owned_paths) {
      // 轻量模式只核对声明路径本身，不遍历其中的文件。
      struct stat extra{};
      if (::lstat(path.c_str(), &extra) != 0)
        throw std::runtime_error("cannot stat extra owned path: " + std::string(std::strerror(errno)));
      if (!S_ISDIR(extra.st_mode)) throw std::runtime_error("extra owned path must be a non-symlink directory");
      if (extra.st_dev != info.st_dev) throw std::runtime_error("owned path crosses the project volume: " + path.string());
    }
    // 只获取卷级可用空间；project 使用量保持未知，不遍历目录树。
    snapshot.available_bytes = fs::space(project).available;
    if (snapshot.available_bytes == std::numeric_limits<std::uintmax_t>::max())
      throw std::runtime_error("volume available space is unknown");
    return snapshot;
  } catch (const std::exception& error) {
    return std::unexpected(error.what());
  }
}

std::expected<Decision, std::string> assess(const Policy& policy, const Snapshot& snapshot,
                                         std::uint64_t peak) {
  try {
    validate_policy(policy);
    Decision result;
    result.requested_peak_bytes = peak;
    if (snapshot.allocated_bytes) {
      result.warning = *snapshot.allocated_bytes >= policy.warning_bytes ||
          (*snapshot.allocated_bytes < policy.warning_bytes && peak >= policy.warning_bytes - *snapshot.allocated_bytes);
    }
    const bool over_project_limit = snapshot.allocated_bytes &&
        (*snapshot.allocated_bytes > policy.project_limit_bytes ||
         peak > policy.project_limit_bytes - std::min(*snapshot.allocated_bytes, policy.project_limit_bytes));
    if (snapshot.available_bytes < policy.min_free_bytes) {
      result.reason = "volume_free_floor_already_violated";
    } else {
      // 项目额度是软预算，只产生提醒；卷剩余空间和本次峰值仍作为写入安全边界。
      const auto volume_room = snapshot.available_bytes - policy.min_free_bytes;
      result.max_growth_bytes = volume_room;
      result.allowed = peak <= result.max_growth_bytes;
      result.reason = !result.allowed ? "volume_peak_exceeded" :
          (!snapshot.allocated_bytes ? "volume_peak_within_floor_project_usage_unknown" :
           (over_project_limit ? "project_budget_exceeded" :
            (result.warning ? "project_budget_warning" : "within_snapshot_budget")));
    }
    return result;
  } catch (const std::exception& error) {
    return std::unexpected(error.what());
  }
}

std::string report_json(const Policy& policy, const Snapshot& snapshot, const Decision& decision) {
  return nlohmann::json{
      {"operation", "read_only_resource_preflight"}, {"allowed", decision.allowed},
      {"warning", decision.warning}, {"reason", decision.reason},
      {"allocated_bytes", snapshot.allocated_bytes ? nlohmann::json(*snapshot.allocated_bytes) : nlohmann::json(nullptr)},
      {"project_usage_measured", snapshot.allocated_bytes.has_value()}, {"available_bytes", snapshot.available_bytes},
      {"requested_peak_bytes", decision.requested_peak_bytes}, {"max_growth_bytes", decision.max_growth_bytes},
      {"policy", {{"project_limit_bytes", policy.project_limit_bytes}, {"warning_bytes", policy.warning_bytes},
                   {"min_free_bytes", policy.min_free_bytes}}},
      {"reservation_created", false}}.dump(2);
}
}  // namespace lab::resources
