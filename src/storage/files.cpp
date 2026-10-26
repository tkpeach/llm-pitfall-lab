// 封装不可替代的 POSIX 文件语义：禁止沿目标符号链接打开、完整读写、fsync 与 SHA-256。
// 路径必须相对已打开目录逐层解析，防止清单路径跳出项目或通过目录链接写到外部。
#include "storage/files.hpp"
#include "lab_build_info.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <unistd.h>
namespace lab::storage {
Fd::~Fd() { if (fd_>=0) ::close(fd_); }
Fd& Fd::operator=(Fd&& other) noexcept {
  if (this!=&other) { if (fd_>=0) ::close(fd_); fd_=std::exchange(other.fd_,-1); }
  return *this;
}
[[noreturn]] void fail(std::string_view operation) {
  throw std::runtime_error(std::string(operation)+": "+std::strerror(errno));
}
Fd directory(const std::filesystem::path& path) {
  const int fd=::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
  if (fd<0) fail("open directory");
  return Fd(fd);
}
Parent parent_at(int root,const std::filesystem::path& relative) {
  if (relative.empty() || relative.is_absolute()) throw std::runtime_error("require a non-empty relative path");
  std::vector<std::string> parts;
  for (const auto& component:relative) {
    auto part=component.string();
    if (part.empty() || part=="." || part==".." || part.find('\0')!=std::string::npos)
      throw std::runtime_error("unsafe path component");
    parts.push_back(std::move(part));
  }
  const int copy=::fcntl(root,F_DUPFD_CLOEXEC,0);
  if (copy<0) fail("duplicate root descriptor");
  Fd current(copy);
  for (std::size_t i=0;i+1<parts.size();++i) {
    const int fd=::openat(current.get(),parts[i].c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
    if (fd<0) fail("open path parent");
    current=Fd(fd);
  }
  return {std::move(current),parts.back()};
}
Fd file_at(int directory,std::string_view name) {
  if (name.empty() || name.find('/')!=std::string_view::npos || name.find('\0')!=std::string_view::npos || name=="." || name=="..")
    throw std::runtime_error("unsafe file name");
  // 先以非阻塞方式打开再验类型，避免损坏目录中的 FIFO 在 fstat 前就让核验永久等待。
  const int fd=::openat(directory,std::string(name).c_str(),O_RDONLY|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC);
  if (fd<0) fail("open file");
  Fd file(fd);
  if (!S_ISREG(stat_fd(fd).st_mode)) throw std::runtime_error("expected regular file");
  return file;
}
struct stat stat_fd(int fd) {
  struct stat value{};
  if (::fstat(fd,&value)!=0) fail("fstat");
  return value;
}
void sync(int fd) { if (::fsync(fd)!=0) fail("fsync"); }
void write_all(int fd,const char* bytes,std::size_t size) {
  while (size) {
    const auto n=::write(fd,bytes,size);
    if (n<0 && errno==EINTR) continue;
    if (n<=0) fail("write");
    bytes+=n;size-=static_cast<std::size_t>(n);
  }
}
std::string read_all(int fd,std::size_t limit) {
  const auto info=stat_fd(fd);
  if (!S_ISREG(info.st_mode) || info.st_size<0 || static_cast<std::uint64_t>(info.st_size)>limit)
    throw std::runtime_error("metadata file is not regular or exceeds size limit");
  if (::lseek(fd,0,SEEK_SET)<0) fail("seek");
  std::string result;
  std::array<char,65536> buffer{};
  for (;;) {
    const auto n=::read(fd,buffer.data(),buffer.size());
    if (n<0 && errno==EINTR) continue;
    if (n<0) fail("read");
    if (!n) break;
    if (static_cast<std::size_t>(n)>limit-result.size()) throw std::runtime_error("metadata grew beyond size limit");
    result.append(buffer.data(),static_cast<std::size_t>(n));
  }
  return result;
}
Fingerprint fingerprint(int fd) {
  const auto before=stat_fd(fd);
  if (!S_ISREG(before.st_mode)) throw std::runtime_error("cannot hash non-regular file");
  if (::lseek(fd,0,SEEK_SET)<0) fail("seek for hash");
  CC_SHA256_CTX context;
  if (!CC_SHA256_Init(&context)) throw std::runtime_error("SHA-256 initialization failed");
  std::array<char,65536> buffer{};
  std::uint64_t bytes=0;
  for (;;) {
    const auto n=::read(fd,buffer.data(),buffer.size());
    if (n<0 && errno==EINTR) continue;
    if (n<0) fail("read for hash");
    if (!n) break;
    bytes+=static_cast<std::uint64_t>(n);
    if (before.st_size<0 || bytes>static_cast<std::uint64_t>(before.st_size))
      throw std::runtime_error("file grew while hashing");
    if (!CC_SHA256_Update(&context,buffer.data(),static_cast<CC_LONG>(n))) throw std::runtime_error("SHA-256 update failed");
  }
  const auto after=stat_fd(fd);
  if (before.st_size!=after.st_size || before.st_mtimespec.tv_sec!=after.st_mtimespec.tv_sec ||
      before.st_mtimespec.tv_nsec!=after.st_mtimespec.tv_nsec || before.st_ctimespec.tv_sec!=after.st_ctimespec.tv_sec ||
      before.st_ctimespec.tv_nsec!=after.st_ctimespec.tv_nsec || bytes!=static_cast<std::uint64_t>(after.st_size))
    throw std::runtime_error("file changed while hashing");
  std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256_Final(digest.data(),&context)) throw std::runtime_error("SHA-256 finalization failed");
  static constexpr char hex[]="0123456789abcdef";
  std::string hash;
  for (auto c:digest) { hash+=hex[c>>4]; hash+=hex[c&15]; }
  return {bytes,hash};
}
std::vector<std::string> entries(int directory_fd) {
  const int copy=::fcntl(directory_fd,F_DUPFD_CLOEXEC,0);
  if (copy<0) fail("duplicate directory");
  auto* raw=::fdopendir(copy);
  if (!raw) { ::close(copy);fail("fdopendir"); }
  std::unique_ptr<DIR,int(*)(DIR*)> dir(raw,::closedir);
  ::rewinddir(raw);
  std::vector<std::string> names;
  errno=0;
  while (auto* entry=::readdir(raw)) {
    const std::string name=entry->d_name;
    if (name!="." && name!="..") names.push_back(name);
    errno=0;
  }
  if (errno) fail("readdir");
  return names;
}
}  // namespace lab::storage
