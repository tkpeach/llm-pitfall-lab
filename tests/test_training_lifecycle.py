#!/usr/bin/env python3
"""Drive separate workers through continuous and pause/new-process/resume paths."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections.abc import Iterator
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import time



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


def pause_worker(command: list[str], ready_file: Path, *, ready_timeout: float = 20,
                 exit_timeout: float = 30) -> tuple[subprocess.CompletedProcess[str], float]:
    with worker(command) as process:
        deadline = time.monotonic() + ready_timeout
        while not ready_file.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if not ready_file.exists():
            raise AssertionError("pause worker never reached rendezvous")
        requested_at = time.monotonic()
        process.send_signal(signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=exit_timeout)
        except subprocess.TimeoutExpired as error:
            raise AssertionError("pause worker did not exit") from error
        latency = time.monotonic() - requested_at
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr), latency


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise AssertionError(f"{label} failed ({result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def main() -> int:
    import mlx.core as mx

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())
    total_steps, pause_step = 8, 3

    with tempfile.TemporaryDirectory(prefix="training-lifecycle-") as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        continuous_file = Path(temporary) / "continuous.safetensors"
        resumed_file = Path(temporary) / "resumed.safetensors"
        ready_file = Path(temporary) / "pause-ready"
        continuous = run([binary, "continuous", str(root), str(continuous_file), str(total_steps)])
        require_success(continuous, "continuous worker")

        pause, pause_latency = pause_worker(
            [binary, "pause", str(root), "step-0003", str(pause_step), str(total_steps), str(ready_file)],
            ready_file,
        )
        require_success(pause, "pause worker")

        checkpoint = root / "runs" / "checkpoints" / "step-0003" / "control.json"
        control = json.loads(checkpoint.read_text())["state"]
        assert control["step"] == pause_step
        assert control["optimizer_step"] == pause_step
        assert control["data_position"] == pause_step

        resumed = run([binary, "resume", str(root), "step-0003", str(resumed_file), str(total_steps)])
        require_success(resumed, "resumed worker")
        continuous_tensors = mx.load(str(continuous_file))
        resumed_tensors = mx.load(str(resumed_file))
        if set(continuous_tensors) != set(resumed_tensors):
            raise AssertionError("continuous and resumed tensor sets differ")
        for name in sorted(continuous_tensors):
            if not bool(mx.array_equal(continuous_tensors[name], resumed_tensors[name]).item()):
                raise AssertionError(f"continuous and resumed tensor differ: {name}")
        if bool(mx.array_equal(continuous_tensors["b"], mx.zeros_like(continuous_tensors["b"])).item()):
            raise AssertionError("trainable adapter b did not change")

        report = {
            "continuous_exit": 0,
            "pause_exit": pause.returncode,
            "pause_worker_exited": pause.returncode is not None,
            "pause_step": pause_step,
            "pause_latency_ms": round(pause_latency * 1000, 3),
            "resume_exit": resumed.returncode,
            "total_steps": total_steps,
            "exact_tensor_matches": sorted(continuous_tensors),
            "adapter_b_updated": True,
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
