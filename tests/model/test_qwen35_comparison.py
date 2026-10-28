#!/usr/bin/env python3
"""Accept the numerical validation capability on the fixed D4 design scenario."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from test_training_lifecycle import run, require_success


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("root", "artifact", "binary", "resource-binary", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    tool = root / "tools/numerical_validation/qwen35/compare.py"
    result = run([sys.executable, str(tool), "--root", str(root),
                  "--artifact", str(args.artifact), "--binary", str(args.binary),
                  "--resource-binary", str(args.resource_binary), "--report", str(args.report),
                  "--input", str(root / "fixtures/qwen35-4b/numerical-input.json")], timeout=840)
    require_success(result, "fixed D4 numerical capability acceptance")
    report = json.loads(args.report.read_text())
    assert report["status"] == "passed"
    assert len(report["input"]["token_ids"]) == 22
    assert set(report["native_workers"]) == set(report["reference_workers"]) == {"0", "1", "3"}
    assert len(report["tensor_metrics"]) == 229
    assert report["tolerances"]["scaled_rmse"] == 0.01
    assert report["tolerances"]["max_logit_absolute_error"] == 0.5
    expected = {"embedding", "final_hidden", "logits", *(f"layer.{i}" for i in range(32))}
    for i in range(32):
        kinds = ("keys", "values") if (i + 1) % 4 == 0 else ("conv", "recurrent")
        expected.update(f"cache.layer.{i}.{kind}" for kind in kinds)
    assert {k.removeprefix("chunk0/") for k in report["tensor_metrics"] if k.startswith("chunk0/")} == expected
    assert all(w["generated"] == [17, 248046] for group in ("native_workers", "reference_workers") for w in report[group].values())
    print("PASS: fixed D4 input, three modes, all layers/states, numerical limits and greedy output")


if __name__ == "__main__":
    main()
