"""Single-worker locking and cooperative pause at explicit safe boundaries."""

import fcntl
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def worker_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "worker.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another worker owns this run directory") from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class PauseControl:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "pause.request"
        self.requested_at = None
        self.previous_handlers = {}

    def __enter__(self):
        def handler(signum, frame):
            if self.requested_at is None:
                self.requested_at = time.time()
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous_handlers[signum] = signal.signal(signum, handler)
        return self

    def __exit__(self, *args):
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)

    def requested(self) -> bool:
        if self.path.exists():
            request = json.loads(self.path.read_text())
            self.requested_at = self.requested_at or request["requested_at"]
            # Consume before saving: any later request remains for the next worker.
            self.path.unlink()
        return self.requested_at is not None


def request_pause(run_dir: Path, guard) -> None:
    if not run_dir.is_dir():
        raise ValueError("run directory does not exist")
    guard.check(65536)
    destination = run_dir / "pause.request"
    temporary = run_dir / f".pause-{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps({"requested_at": time.time()}))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
