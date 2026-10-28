# POSIX 信号与 `sigaction`

## 信号是通知，不是函数调用

信号是操作系统向进程或线程报告事件的一种机制。终端通常把 Ctrl+C 转成 SIGINT 发给前台进程组；其他进程可请求向目标进程发送 SIGTERM；某些硬件异常则会关联到发生异常的线程。

每种信号都有一个处理规则，称为 disposition：执行默认动作、忽略，或运行程序注册的处理函数。`sigaction()` 用来查询或更改这个规则；它不负责发送信号。SIGKILL 和 SIGSTOP 不能被捕获、忽略或屏蔽。

POSIX 进程收到可递送的信号时，系统会转到程序的处理函数运行。处理函数返回后，程序通常从被打断的执行上下文继续。被屏蔽的信号会保持 pending，直到可以递送。信号递送类似异步中断，不应把它当作普通代码在某一行主动调用了回调。参见 [Apple `sigaction` 手册](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/sigaction.2.html) 和 [POSIX `sigaction` 规范](https://pubs.opengroup.org/onlinepubs/9799919799/functions/sigaction.html)。

## 注册处理规则

```c
#include <signal.h>
#include <stdio.h>

static volatile sig_atomic_t stop_requested = 0;

static void on_sigint(int signo) {
    (void)signo;
    stop_requested = 1;
}

int main(void) {
    struct sigaction action = {0};
    struct sigaction previous_action;

    action.sa_handler = on_sigint;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;

    if (sigaction(SIGINT, &action, &previous_action) == -1) {
        perror("sigaction");
        return 1;
    }

    while (!stop_requested) {
        /* 执行一段可安全完成的工作 */
    }

    /* 普通程序流程中的清理、保存和退出 */
    return 0;
}
```

`sigaction(signum, act, oldact)` 中，`signum` 是信号编号，`act` 是新规则，`oldact` 非空时接收旧规则。成功返回 0；失败返回 -1 并设置 `errno`。新规则通常会持续生效，直到再次修改或进程执行新程序映像。

## `struct sigaction` 的字段

| 字段 | 作用 |
| --- | --- |
| `sa_handler` | 普通处理函数，形如 `void handler(int)`；也可以设成 `SIG_DFL` 或 `SIG_IGN` |
| `sa_sigaction` | 设置 `SA_SIGINFO` 时使用的处理函数，可取得 `siginfo_t` 等信号来源信息；不要同时设置 `sa_handler` 和 `sa_sigaction` |
| `sa_mask` | 处理函数运行期间额外屏蔽的信号集合。触发处理函数的信号通常也会自动屏蔽，除非设置 `SA_NODEFER` |
| `sa_flags` | 选择附加行为，例如 `SA_SIGINFO`、`SA_RESTART`、`SA_ONSTACK`、`SA_NODEFER` 和 `SA_RESETHAND`；可用选项及细节应查目标系统手册 |

`sigemptyset(&action.sa_mask)` 表示不额外屏蔽其他信号；它不是“屏蔽所有信号”。`SA_RESTART` 只会让部分可重启的系统调用在处理函数返回后重试，并非所有被打断的调用都会自动重做。若没有设置它，某些调用可能以 `EINTR` 返回，调用者要按 API 约定处理。

## 为什么处理函数要短

信号可能在程序更新数据结构、持有锁或执行库函数时到达。处理函数会插入这段执行过程；若它调用了同一套非重入代码，可能观察到中间状态、破坏内部数据或死锁。

例如，主程序调用 `printf()` 更新标准输出缓冲区时，若处理函数也调用 `printf()`，两个调用可能重入同一份尚未更新完成的缓冲状态。`malloc()`/`free()` 也可能被内部堆结构或锁打断。`std::mutex`、异常、日志框架和 ML/GPU 库都不适合从异步处理函数直接调用。

POSIX 将一组接口定义为 async-signal-safe；普通库函数不因“简单”就自动安全。典型处理函数只设置标志并返回，保存、日志、释放复杂资源等操作留给普通控制流。[Linux `signal-safety(7)` 对缓冲 I/O 与非重入调用的说明](https://man7.org/linux/man-pages/man7/signal-safety.7.html)；[POSIX async-signal-safe 规则](https://pubs.opengroup.org/onlinepubs/9799919799/functions/V2_chap02.html)。

简单、单线程的通知标志可以使用 `volatile sig_atomic_t`。它只适合处理信号与正常代码之间的有限标志传递，不是一般多线程共享状态的同步原语。信号处理函数里也不应访问需要锁保护的普通共享对象。

## 多线程进程的信号

信号 disposition 通常是进程共享的，但每个线程有自己的信号屏蔽集合。发给整个进程的信号可递送给任一未屏蔽该信号的线程；若多个线程都可接收，具体选择由系统决定。线程信号屏蔽和进程/线程定向信号的区别见 [Linux `signal(7)`](https://man7.org/linux/man-pages/man7/signal.7.html)。

因此，多线程程序不应假定处理函数一定在主线程运行。常见的可控模式是：在创建工作线程前屏蔽目标信号，再由专门线程使用 `sigwait()` 同步接收；随后该线程可以在普通线程流程中更新状态或通知其他线程。Linux 的 `signalfd` 是另一种 Linux 专有选择，不属于通用 POSIX 接口。

标准信号不是可靠计数器：同一种标准信号在尚未递送时重复到达，通常只保留一个 pending 状态；若需要排队多个事件，应评估实时信号或其他 IPC，而不是数 Ctrl+C 次数。

## 常见执行顺序

```text
注册：程序 → sigaction() → 系统记录该信号的 disposition
发送：终端/其他进程/系统事件 → 信号成为目标的 pending 信号
递送：信号未被屏蔽 → 系统安排处理函数运行
返回：处理函数结束 → 恢复被中断的执行上下文
后续：普通程序检查标志 → 完成安全边界工作
```

处理函数只适合做非常小的异步响应。暂停训练、保存检查点、刷新复杂日志或安全退出，应在训练循环的完整更新边界处理。

本仓库的通用实现见 [runner 控制模块](../../src/runner/control.cpp)：处理函数只设置 `sig_atomic_t` 标志；runner 在完整更新回调返回后检查标志，再执行检查点回调。合成连续/暂停/恢复验证见 [生命周期测试](../../tests/integration/training_lifecycle.cpp)。

## 复习问题

- `sigaction()` 更改什么？它会不会自己发送信号？
- `sa_mask` 与 `sa_flags` 分别控制什么？当前信号为何通常会自动屏蔽？
- `printf()` 或获取互斥锁为什么不适合放在异步信号处理函数中？
- 多线程进程收到 process-directed signal 时，为什么不能假定主线程会处理它？
- 需要保存检查点时，为什么更适合让处理函数设置标志、训练循环在安全边界保存？
