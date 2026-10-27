#!/usr/bin/env python3
"""Verify lifecycle-driver failure cleanup with real, GPU-free child processes."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_training_lifecycle as driver


class WorkerCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workers = []
        real_popen = subprocess.Popen

        def spawn(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            self.workers.append(process)
            return process

        self.patcher = patch.object(driver.subprocess, "Popen", side_effect=spawn)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.cleanup_workers)

    def cleanup_workers(self) -> None:
        # Keep a failing regression from leaking the worker it is testing.
        for process in self.workers:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def assert_reaped(self) -> None:
        self.assertEqual(len(self.workers), 1)
        self.assertIsNotNone(self.workers[0].returncode)
        self.assertIsNotNone(self.workers[0].poll())

    def test_missing_ready_file_reaps_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, "never reached rendezvous"):
                driver.pause_worker(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    Path(directory) / "ready", ready_timeout=0.2, exit_timeout=0.2,
                )
            self.assert_reaped()

    def test_ignored_pause_signal_reaps_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            script = (
                "import pathlib, signal, sys, time; "
                "signal.signal(signal.SIGINT, signal.SIG_IGN); "
                "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
            )
            with self.assertRaisesRegex(AssertionError, "did not exit"):
                driver.pause_worker([sys.executable, "-c", script, str(ready)], ready,
                                    ready_timeout=5, exit_timeout=0.2)
            self.assert_reaped()

    def test_unexpected_exception_reaps_worker(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "driver failure"):
            with driver.worker([sys.executable, "-c", "import time; time.sleep(60)"]):
                raise RuntimeError("driver failure")
        self.assert_reaped()

    def test_continuous_or_resume_timeout_reaps_worker(self) -> None:
        with self.assertRaises(subprocess.TimeoutExpired):
            driver.run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.2)
        self.assert_reaped()


if __name__ == "__main__":
    unittest.main()
