#include "lab_build_info.hpp"
#include "task/evaluator.hpp"

#include <cstdio>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
std::string read_input(const char* path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error(std::string("cannot open input: ") + path);
  const auto size = input.tellg();
  // 在分配读取缓冲前限制小批量输入大小，避免误读大文件；这不替代 D2 的统一资源保护。
  constexpr std::streamoff limit = 16 * 1024 * 1024;
  if (size < 0 || size > limit)
    throw std::runtime_error("input is unreadable or exceeds 16 MiB");
  std::string contents(static_cast<std::size_t>(size), '\0');
  input.seekg(0);
  if (!input.read(contents.data(), static_cast<std::streamsize>(size)))
    throw std::runtime_error(std::string("cannot read input: ") + path);
  return contents;
}
}  // namespace

int main(int argc, char** argv) {
  const auto help = "Usage: pitfall-lab --version | --help\n"
                    "       pitfall-lab evaluate --gold GOLD.json --predictions PREDICTIONS.json";
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::println("{}", help);
    return 0;
  }
  if (argc == 6 && std::string_view(argv[1]) == "evaluate" &&
      std::string_view(argv[2]) == "--gold" && std::string_view(argv[4]) == "--predictions") {
    // CLI 只连接文件输入、任务模块和报告输出；不在这里解析/修复答案，也不启动 GPU。
    try {
      auto report = lab::task::evaluate_batch(read_input(argv[3]), read_input(argv[5]));
      if (!report) {
        // 评估材料不合法与“答案得零分”不同：前者退出报错，后者仍正常生成评分报告。
        std::println(stderr, "Evaluation input error: {}", report.error());
        return 1;
      }
      // stdout 保持为可供工具读取的 JSON；故障诊断走 stderr，避免混入报告。
      std::println("{}", lab::task::report_json(*report));
      return 0;
    } catch (const std::exception& error) {
      std::println(stderr, "Evaluation failed: {}", error.what());
      return 1;
    }
  }
  if (argc == 2 && std::string_view(argv[1]) == "--version") {
    std::println("pitfall-lab {} (C++23, MLX SDK {})", lab::build::version, lab::build::mlx_version);
    return 0;
  }
  std::println(stderr, "{}", help);
  return 2;
}
