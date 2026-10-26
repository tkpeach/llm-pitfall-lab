// 固定 macOS 文件原语：描述符生命周期、安全相对路径和内容摘要，供受控写入/检查点共用。
// 本层不判断资源额度或训练状态；失败抛异常，由业务接口转换为显式错误。
#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <utility>
#include <sys/stat.h>
namespace lab::storage {
class Fd {
  int fd_ = -1;
public:
  explicit Fd(int fd=-1) noexcept : fd_(fd) {}
  ~Fd();
  Fd(Fd&& other) noexcept : fd_(std::exchange(other.fd_,-1)) {}
  Fd& operator=(Fd&& other) noexcept;
  Fd(const Fd&)=delete;
  Fd& operator=(const Fd&)=delete;
  int get() const noexcept { return fd_; }
};
struct Parent { Fd directory; std::string name; };
struct Fingerprint { std::uint64_t bytes; std::string sha256; };
[[noreturn]] void fail(std::string_view operation);
Fd directory(const std::filesystem::path& path);
Parent parent_at(int root, const std::filesystem::path& relative);
Fd file_at(int directory, std::string_view name);
struct stat stat_fd(int fd);
void sync(int fd);
void write_all(int fd, const char* bytes, std::size_t size);
std::string read_all(int fd, std::size_t limit);
Fingerprint fingerprint(int fd);
std::vector<std::string> entries(int directory);
}  // namespace lab::storage
