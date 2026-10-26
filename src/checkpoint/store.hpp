// 版本化检查点的保存与核验接口：控制状态用 JSON，张量保持独立文件，整体原子发布。
// 这里只保证容器完整性与运行身份；哪些训练状态必须保存由后续运行控制/模型模块负责。
#pragma once
#include "resources/preflight.hpp"
#include <map>
namespace lab::checkpoint {
struct Identity {
  std::string model, data, template_id, configuration;
  bool operator==(const Identity&) const = default;
};
struct InputFile { std::string name; std::filesystem::path source; };
struct Artifact { std::string name; std::uint64_t bytes; std::string sha256; };
struct Request {
  std::string name;
  Identity identity;
  std::string control_json;
  std::vector<InputFile> files;
  std::uint64_t peak_bytes;
};
struct Verified {
  std::filesystem::path directory;
  Identity identity;
  std::string control_json;
  std::vector<Artifact> artifacts;
};
std::expected<Verified,std::string> save(const std::filesystem::path& root,
                                      const resources::Policy& policy,const Request& request);
std::expected<Verified,std::string> verify(const std::filesystem::path& root,std::string_view name,
                                        const Identity& expected);
// CLI 输入/报告与容器格式一起维护，不让命令入口复制检查点校验规则。
std::expected<Request,std::string> parse_request(std::string_view text);
std::expected<Identity,std::string> parse_identity(std::string_view text);
std::string report_json(const Verified& checkpoint);
}  // namespace lab::checkpoint
