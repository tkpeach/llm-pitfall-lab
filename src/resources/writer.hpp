// 提供检查点暂存目录、文件写入和不覆盖发布的轻量流程；磁盘盘点由阶段性 preflight 负责。
#pragma once
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
namespace lab::resources {
class WriteSession {
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit WriteSession(std::unique_ptr<Impl> impl);
public:
  ~WriteSession();
  WriteSession(WriteSession&&) noexcept;
  WriteSession& operator=(WriteSession&&) noexcept;
  WriteSession(const WriteSession&)=delete;
  WriteSession& operator=(const WriteSession&)=delete;
  static std::expected<WriteSession,std::string> begin(const std::filesystem::path& root);
  const std::filesystem::path& root() const;
  // 目录必须逐层建立；ensure 允许已有真实目录，create 要求目标此前不存在。
  std::expected<void,std::string> ensure_directory(const std::filesystem::path& relative);
  std::expected<void,std::string> create_directory(const std::filesystem::path& relative);
  std::expected<void,std::string> write_file(const std::filesystem::path& relative,std::string_view bytes);
  std::expected<void,std::string> copy_file(const std::filesystem::path& source,
                                         const std::filesystem::path& relative);
  // 发布使用同卷、不覆盖的原子重命名；目录同步失败可能已可见，调用者须核验而不能覆盖重试。
  std::expected<void,std::string> publish(const std::filesystem::path& stage,const std::filesystem::path& target);
  // 只清理本会话创建且身份未变的目录，不删除先前运行或用户已有目录。
  void discard_directory(const std::filesystem::path& relative) noexcept;
};
}  // namespace lab::resources
