// 提供检查点暂存目录、文件写入和不覆盖发布的轻量流程；资源盘点由阶段性 preflight 负责。
#include "resources/writer.hpp"
#include "storage/files.hpp"
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <map>
#include <stdexcept>
#include <sys/stdio.h>
#include <unistd.h>

namespace lab::resources {
namespace fs=std::filesystem;
namespace io=lab::storage;

struct WriteSession::Impl {
  fs::path project;
  io::Fd directory;
  bool failed=false;
  std::map<fs::path,std::pair<dev_t,ino_t>> created_directories;

  template<class F> std::expected<void,std::string> run(F&& action) {
    try {
      if (failed) throw std::runtime_error("write session failed; start a new session");
      action();
      return {};
    } catch (const std::exception& error) {
      failed=true;
      return std::unexpected(error.what());
    }
  }

  void mkdir(const fs::path& relative,bool allow_existing) {
    auto parent=io::parent_at(directory.get(),relative);
    struct stat existing{};
    if (::fstatat(parent.directory.get(),parent.name.c_str(),&existing,AT_SYMLINK_NOFOLLOW)==0) {
      if (allow_existing && S_ISDIR(existing.st_mode)) return;
      throw std::runtime_error("directory target already exists or is not a real directory");
    }
    if (errno!=ENOENT) io::fail("inspect directory target");
    if (::mkdirat(parent.directory.get(),parent.name.c_str(),0700)!=0) io::fail("create directory");
    if (::fstatat(parent.directory.get(),parent.name.c_str(),&existing,AT_SYMLINK_NOFOLLOW)!=0)
      io::fail("stat created directory");
    created_directories.emplace(relative,std::pair{existing.st_dev,existing.st_ino});
    io::sync(parent.directory.get());
  }

  template<class Producer> void file(const fs::path& relative,Producer&& producer) {
    auto parent=io::parent_at(directory.get(),relative);
    const int fd=::openat(parent.directory.get(),parent.name.c_str(),
        O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if (fd<0) io::fail("create new controlled file");
    io::Fd output(fd);
    const auto identity=io::stat_fd(fd);
    try {
      auto sink=[&](const char* data,std::size_t size) { io::write_all(fd,data,size); };
      producer(sink);
      io::sync(fd);
    } catch (...) {
      struct stat current{};
      if (::fstatat(parent.directory.get(),parent.name.c_str(),&current,AT_SYMLINK_NOFOLLOW)==0 &&
          current.st_dev==identity.st_dev && current.st_ino==identity.st_ino)
        ::unlinkat(parent.directory.get(),parent.name.c_str(),0);
      throw;
    }
  }
};

WriteSession::WriteSession(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
WriteSession::~WriteSession()=default;
WriteSession::WriteSession(WriteSession&&) noexcept=default;
WriteSession& WriteSession::operator=(WriteSession&&) noexcept=default;

std::expected<WriteSession,std::string> WriteSession::begin(const fs::path& root) {
  try {
    auto impl=std::make_unique<Impl>();
    impl->project=fs::canonical(root);
    impl->directory=io::directory(impl->project);
    return WriteSession(std::move(impl));
  } catch (const std::exception& error) {
    return std::unexpected(error.what());
  }
}

const fs::path& WriteSession::root() const { return impl_->project; }
std::expected<void,std::string> WriteSession::ensure_directory(const fs::path& relative) {
  return impl_->run([&]{impl_->mkdir(relative,true);});
}
std::expected<void,std::string> WriteSession::create_directory(const fs::path& relative) {
  return impl_->run([&]{impl_->mkdir(relative,false);});
}
std::expected<void,std::string> WriteSession::write_file(const fs::path& relative,std::string_view bytes) {
  return impl_->run([&]{impl_->file(relative,[&](auto&& sink){sink(bytes.data(),bytes.size());});});
}
std::expected<void,std::string> WriteSession::copy_file(const fs::path& source,const fs::path& relative) {
  return impl_->run([&]{
    if (!fs::is_regular_file(fs::symlink_status(source)))
      throw std::runtime_error("copy source must be a regular non-symlink file");
    auto input=io::file_at(io::directory(source.parent_path().empty()?fs::path("."):source.parent_path()).get(),
                           source.filename().string());
    const auto before=io::stat_fd(input.get());
    impl_->file(relative,[&](auto&& sink){
      std::array<char,65536> buffer{};
      for (;;) {
        const auto n=::read(input.get(),buffer.data(),buffer.size());
        if (n<0 && errno==EINTR) continue;
        if (n<0) io::fail("read source for controlled copy");
        if (!n) break;
        sink(buffer.data(),static_cast<std::size_t>(n));
      }
      const auto after=io::stat_fd(input.get());
      if (before.st_size!=after.st_size || before.st_mtimespec.tv_sec!=after.st_mtimespec.tv_sec ||
          before.st_mtimespec.tv_nsec!=after.st_mtimespec.tv_nsec || before.st_ctimespec.tv_sec!=after.st_ctimespec.tv_sec ||
          before.st_ctimespec.tv_nsec!=after.st_ctimespec.tv_nsec)
        throw std::runtime_error("copy source changed during checkpoint capture");
    });
  });
}
std::expected<void,std::string> WriteSession::publish(const fs::path& stage,const fs::path& target) {
  return impl_->run([&]{
    const auto owned=impl_->created_directories.find(stage);
    if (owned==impl_->created_directories.end()) throw std::runtime_error("stage was not created by this session");
    auto from=io::parent_at(impl_->directory.get(),stage);
    auto to=io::parent_at(impl_->directory.get(),target);
    const int fd=::openat(from.directory.get(),from.name.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
    if (fd<0) io::fail("open stage for publication");
    io::Fd staged(fd);
    const auto info=io::stat_fd(fd);
    if (std::pair{info.st_dev,info.st_ino}!=owned->second) throw std::runtime_error("stage identity changed");
    io::sync(fd);
    if (::renameatx_np(from.directory.get(),from.name.c_str(),to.directory.get(),to.name.c_str(),RENAME_EXCL)!=0)
      io::fail("publish without replacing existing checkpoint");
    try { io::sync(to.directory.get());io::sync(from.directory.get()); }
    catch (const std::exception& error) {
      throw std::runtime_error(std::string("checkpoint visible but directory sync failed: ")+error.what());
    }
  });
}
void WriteSession::discard_directory(const fs::path& relative) noexcept {
  try {
    const auto found=impl_->created_directories.find(relative);
    if (found==impl_->created_directories.end()) return;
    auto parent=io::parent_at(impl_->directory.get(),relative);
    struct stat current{};
    if (::fstatat(parent.directory.get(),parent.name.c_str(),&current,AT_SYMLINK_NOFOLLOW)==0 &&
        S_ISDIR(current.st_mode) && std::pair{current.st_dev,current.st_ino}==found->second)
      fs::remove_all(impl_->project/relative);
  } catch (...) { /* 清理失败时保留目录，交由阶段性资源盘点发现。 */ }
}
}  // namespace lab::resources
