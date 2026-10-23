#include "lab_build_info.hpp"

#include <cstdio>
#include <print>
#include <string_view>

int main(int argc, char** argv) {
  const auto help = "Usage: pitfall-lab --version | --help";
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::println("{}", help);
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--version") {
    std::println("pitfall-lab {} (C++23, MLX SDK {})", lab::build::version, lab::build::mlx_version);
    return 0;
  }
  std::println(stderr, "{}", help);
  return 2;
}
