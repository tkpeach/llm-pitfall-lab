"""Run bounded numerical workers and always terminate/reap them on errors."""
from contextlib import contextmanager
from collections.abc import Iterator
import subprocess


@contextmanager
def worker(command: list[str]) -> Iterator[subprocess.Popen[str]]:
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        yield process
    finally:
        # SIGINT/SIGTERM may be deferred or ignored by a stuck worker. Failure
        # cleanup must release its resources and reap it before removing files.
        if process.poll() is None:
            process.kill()
        process.communicate()


def run(command: list[str], *, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    with worker(command) as process:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise AssertionError(f"{label} failed ({result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
