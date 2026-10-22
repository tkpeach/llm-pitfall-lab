"""Small MLX training for lifecycle verification, never a Qwen fallback."""

import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

from .checkpoint import canonical, load_checkpoint, save_checkpoint
from .runner import PauseControl, worker_lock


def _pack(tree):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    result = []
    for key, value in tree_flatten(tree):
        if isinstance(value, mx.array):
            mx.eval(value)
            result.append({"key": key, "dtype": str(value.dtype).split(".")[-1],
                           "shape": list(value.shape), "value": value.tolist()})
        else:
            result.append({"key": key, "dtype": None, "value": value})
    return result


def _unpack(packed):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    result = []
    for leaf in packed:
        value = leaf["value"]
        if leaf["dtype"] is not None:
            if leaf["dtype"] not in ("float32", "int32", "uint32", "int64", "uint64", "bool_"):
                raise ValueError("unsupported mechanism checkpoint dtype")
            value = mx.array(value, dtype=getattr(mx, leaf["dtype"])).reshape(leaf["shape"])
        result.append((leaf["key"], value))
    return tree_unflatten(result)


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def train(root: Path, run_dir: Path, steps: int, seed: int, resume: bool,
          pause_after: int | None, guard) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    if steps < 1 or (pause_after is not None and not 0 <= pause_after < steps):
        raise ValueError("invalid steps or pause boundary")
    if not mx.metal.is_available():
        raise RuntimeError("Metal required; no CPU fallback")
    mx.set_default_device(mx.gpu)
    source_dir = Path(__file__).parent
    code_hash = hashlib.sha256(b"".join((source_dir / name).read_bytes() for name in
        ("mechanism.py", "runner.py", "checkpoint.py"))).hexdigest()
    identity = {"kind": "linear-regression-mechanism-v1", "steps": steps,
                "seed": seed, "learning_rate": 0.02, "code_sha256": code_hash,
                "mlx": importlib.metadata.version("mlx")}
    guard.check(2 * 1024**2)

    with worker_lock(run_dir), PauseControl(run_dir) as control:
        checkpoint = run_dir / "checkpoint.json"
        if resume and not checkpoint.exists():
            raise ValueError("resume requires a valid checkpoint; no fresh-start fallback")
        if not resume and checkpoint.exists():
            raise ValueError("checkpoint exists; use explicit --resume or a new run directory")
        mx.random.seed(seed)
        # Every stochastic training operation takes this explicit PRNG key.
        # Initialization uses the seed only before the saved weights are restored.
        random_key = mx.random.key(seed + 1)
        data_rng = random.Random(seed)
        model = nn.Linear(3, 1)
        optimizer = optim.Adam(learning_rate=0.02)
        # Fixed synthetic dataset. Sampling and input noise use separate saved RNGs.
        x = mx.array([[(i % 5) - 2.0, (i % 7) / 3 - 1, (i % 3) - 1.0] for i in range(17)])
        y = x @ mx.array([[2.0], [-0.5], [3.0]]) + 0.25
        step, consumed, losses = 0, [], []
        if resume:
            state = load_checkpoint(checkpoint, identity)
            step = state["step"]
            if type(step) is not int or not 0 <= step <= steps:
                raise ValueError("invalid checkpoint progress")
            model.update(_unpack(state["model"]))
            optimizer.state = _unpack(state["optimizer"])
            random_key = _unpack(state["mlx_random_key"])["key"]
            data_rng.setstate(_tuples(state["data_random"]))
            consumed, losses = state["consumed_ids"], state["losses"]
            if len(consumed) != step or len(losses) != step:
                raise ValueError("inconsistent data/update progress")
        mx.eval(model.state, optimizer.state, random_key, x, y)

        def objective(model, batch_x, batch_y):
            return mx.mean(mx.square(model(batch_x) - batch_y))

        value_and_grad = nn.value_and_grad(model, objective)
        start = time.perf_counter()
        paused = False

        def persist(status):
            mx.synchronize()
            state = {"step": step, "model": _pack(model.parameters()),
                     "optimizer": _pack(optimizer.state), "mlx_random_key": _pack({"key": random_key}),
                     "data_random": data_rng.getstate(), "consumed_ids": consumed,
                     "losses": losses, "status": status}
            save_checkpoint(checkpoint, identity, state, guard)

        while step < steps:
            if control.requested() or (pause_after is not None and step >= pause_after):
                persist("paused")
                paused = True
                break
            ids = data_rng.sample(range(17), 4)
            keys = mx.random.split(random_key, num=2)
            random_key = keys[0]
            batch_x = x[mx.array(ids)] + 0.01 * mx.random.normal((4, 3), key=keys[1])
            loss, grads = value_and_grad(model, batch_x, y[mx.array(ids)])
            optimizer.update(model, grads)
            mx.eval(model.state, optimizer.state, loss, random_key)
            if not mx.isfinite(loss).item():
                raise RuntimeError("nonfinite loss; no checkpoint committed for this update")
            step += 1
            consumed.append(ids)
            losses.append(loss.item())
            if step % 4 == 0:
                persist("running")
        if not paused:
            if control.requested():
                # All requested updates finished: persist completion rather than repeat work.
                pass
            persist("completed")
        report = {"scope": "mechanism-only; not Qwen training or model performance",
                  "status": "paused" if paused else "completed", "step": step,
                  "pid": os.getpid(), "device": str(mx.default_device()),
                  "seconds": time.perf_counter() - start,
                  "pause_save_latency_seconds": None if control.requested_at is None else
                      max(0, time.time() - control.requested_at),
                  "mlx_peak_allocated_bytes": mx.get_peak_memory()}
        guard.check(len(canonical(report)) + 65536)
        (run_dir / "report.json").write_bytes(canonical(report))
        return report
