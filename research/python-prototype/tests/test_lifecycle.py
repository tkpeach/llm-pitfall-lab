"""Opt-in Metal integration: independent processes, complete state comparison."""

import json
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(os.environ.get("RUN_MLX_TESTS") == "1", "requires Metal GPU access")
class Lifecycle(unittest.TestCase):
    def invoke(self, *arguments, success=True):
        result = subprocess.run([sys.executable, "-m", "llm_pitfall_lab.cli", *arguments],
                                cwd=ROOT, text=True, capture_output=True, timeout=120)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def test_continuous_matches_pause_resume_and_worker_exits(self):
        prefix = f"runs/validation-{uuid.uuid4().hex[:10]}"
        self.invoke("mechanism", "--run", prefix + "/continuous")
        first = self.invoke("mechanism", "--run", prefix + "/resumed", "--pause-after-step", "5")
        self.assertEqual((first["status"], first["step"]), ("paused", 5))
        # subprocess.run has waited for exit: the paused worker is no longer resident.
        self.assertFalse(subprocess.run(["/bin/kill", "-0", str(first["pid"])],
                                       capture_output=True).returncode == 0)
        self.invoke("mechanism", "--run", prefix + "/resumed", "--resume")
        states = []
        for name in ("continuous", "resumed"):
            payload = json.loads((ROOT / prefix / name / "checkpoint.json").read_text())["payload"]
            states.append(payload)
        self.assertEqual(states[0], states[1], "parameters/optimizer/RNG/data/update history differ")

    def test_real_pause_request_and_missing_resume_fail_closed(self):
        prefix = f"runs/validation-{uuid.uuid4().hex[:10]}"
        run_dir = ROOT / prefix
        run_dir.mkdir(parents=True)
        self.invoke("pause", "--run", prefix)
        report = self.invoke("mechanism", "--run", prefix)
        self.assertEqual((report["status"], report["step"]), ("paused", 0))
        self.invoke("mechanism", "--run", prefix, "--resume")
        error = self.invoke("mechanism", "--run", prefix + "-missing", "--resume", success=False)
        self.assertIn("requires a valid checkpoint", error)


if __name__ == "__main__":
    unittest.main()
