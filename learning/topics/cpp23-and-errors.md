# C++23 与错误返回

## `std::expected` 是类型，不是新的语法

`std::expected<T, E>` 是 C++23 标准库模板，使用普通的 C++ 模板语法，声明在 `<expected>` 中。它表示一次操作的结果：成功时保存 `T`，失败时保存 `E`，两者只存在一种。

```cpp
#include <expected>
#include <string>

std::expected<int, std::string> parse_positive(int input) {
  if (input <= 0) {
    return std::unexpected(std::string{"必须是正数"});
  }
  return input;
}

auto result = parse_positive(-1);
if (result) {
  // *result 是成功值。
} else {
  // result.error() 是错误说明。
}
```

`std::unexpected` 用于明确构造失败结果。`std::expected` 让失败成为函数返回类型的一部分，但不会强制调用者处理错误，也不意味着函数绝不会抛异常。

| 方式 | 适用情形与区别 |
| --- | --- |
| `std::optional<T>` | 只有“有值/无值”，无须表达失败原因 |
| `std::expected<T, E>` | 正常控制流中可能失败，需要携带原因并显式处理 |
| 异常 | 通过调用栈传播错误；调用者不在每层返回类型中显式携带错误 |

选择取决于接口和错误传播方式，不必将所有错误统一改成 `expected`。调用 `.value()` 时若处于失败状态仍会抛出异常；应先判断或有意处理这一行为。

## 为什么 C++23 不等于所有特性都可用

语言标准、编译器和标准库是不同层次：编译器处理语法，标准库提供模板和函数。开启 C++23 模式，不保证配套标准库已实现全部 C++23 功能。

`std::expected` 提供结果类型，`std::println` 提供格式化输出。项目选择 C++23 是为了使用合适的现代语言和库能力，但具体能力仍需固定工具链验证；也没有必要为了“使用新标准”重写已有代码。

对固定环境的小项目，声明标准要求、禁止悄悄降级、编译链接正式代码，通常已经足够。只有需要提前诊断特定兼容问题或选择实现路径时，才值得增加单独的特性探测；逐项探测所有标准特性既不现实，也不证明完整支持。

`CheckCXXSourceCompiles` 是 CMake 自带模块，提供试编译并链接小程序的命令，不执行程序。它是可选的配置检查手段，不是 C++23 支持的认证。本项目的实际使用要求见 [构建约定](../../docs/build.md)。

## 复习问题

- `std::expected<int, std::string>` 与 `std::optional<int>` 相比，多表达了什么？
- 既然返回类型是 `expected`，为什么调用 `.value()` 仍可能抛异常？
- 为什么编译器接受 C++23 模式，仍可能找不到某个标准库接口？
