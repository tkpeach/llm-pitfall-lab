"""Explicit commands for contract, resources and lifecycle validation."""

import argparse
import json
from pathlib import Path

from .resources import ResourceGuard


def _run_directory(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    # All worker artifacts live under the same budgeted runs directory.
    if path == root / "runs" or not path.is_relative_to(root / "runs"):
        raise ValueError("run directory must be a child of project runs/")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    resources = commands.add_parser("preflight")
    resources.add_argument("--peak-mib", type=int, required=True)
    score = commands.add_parser("score")
    score.add_argument("--prediction", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True)
    worker = commands.add_parser("mechanism")
    worker.add_argument("--run", required=True)
    worker.add_argument("--steps", type=int, default=12)
    worker.add_argument("--seed", type=int, default=42)
    worker.add_argument("--resume", action="store_true")
    worker.add_argument("--pause-after-step", type=int)
    pause = commands.add_parser("pause")
    pause.add_argument("--run", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        guard = ResourceGuard(root)
        if args.command == "preflight":
            result = guard.check(args.peak_mib * 1024**2)
        elif args.command == "score":
            from .task import score_answer
            result = score_answer(args.prediction.read_text(), json.loads(args.gold.read_text()))
        elif args.command == "pause":
            from .runner import request_pause
            request_pause(_run_directory(root, args.run), guard)
            result = {"status": "pause_requested", "note": "worker exits after a safe save boundary"}
        else:
            from .mechanism import train
            result = train(root, _run_directory(root, args.run), args.steps, args.seed,
                           args.resume, args.pause_after_step, guard)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
