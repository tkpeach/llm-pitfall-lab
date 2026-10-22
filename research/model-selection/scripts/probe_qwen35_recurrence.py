"""Probe real Qwen3.5 recurrence dimensions, not a pretrained model or full training."""

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm.models import gated_delta


def probe(length: int) -> dict:
    mx.random.seed(42)
    q = mx.random.normal((1, length, 16, 128)).astype(mx.bfloat16)
    k = mx.random.normal((1, length, 16, 128)).astype(mx.bfloat16)
    q, k = gated_delta.normalize_qk(q, k, inv_scale=128**-0.5, eps=1e-6)
    v = (0.1 * mx.random.normal((1, length, 32, 128))).astype(mx.bfloat16)
    a = mx.zeros((1, length, 32), dtype=mx.bfloat16)
    b = mx.zeros_like(a)
    a_log = mx.zeros((32,), dtype=mx.float32)
    dt_bias = mx.zeros((32,), dtype=mx.float32)
    mx.eval(q, k, v, a, b, a_log, dt_bias)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()

    start = time.perf_counter()
    out_kernel, state_kernel = gated_delta.gated_delta_update(
        q, k, v, a, b, a_log, dt_bias, use_kernel=True
    )
    out_ops, state_ops = gated_delta.gated_delta_update(
        q, k, v, a, b, a_log, dt_bias, use_kernel=False
    )
    output_error = mx.max(mx.abs(out_kernel.astype(mx.float32) - out_ops.astype(mx.float32)))
    state_error = mx.max(mx.abs(state_kernel - state_ops))
    mx.eval(out_kernel, out_ops, state_kernel, state_ops, output_error, state_error)
    mx.synchronize()
    forward_seconds = time.perf_counter() - start
    if output_error.item() > 1e-3 or state_error.item() > 1e-3:
        raise RuntimeError("Recurrence paths differ beyond this probe's absolute tolerance.")
    del out_kernel, out_ops, state_kernel, state_ops
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    active_before = mx.get_active_memory()

    def objective(q, k, v, a, b):
        out, _ = gated_delta.gated_delta_update(
            q, k, v, a, b, a_log, dt_bias, use_kernel=False
        )
        return mx.mean(mx.square(out.astype(mx.float32)))

    start = time.perf_counter()
    loss, grads = mx.value_and_grad(objective, argnums=(0, 1, 2, 3, 4))(q, k, v, a, b)
    finite = [mx.all(mx.isfinite(grad)) for grad in grads]
    mx.eval(loss, grads, finite)
    mx.synchronize()
    seconds = time.perf_counter() - start
    if not all(value.item() for value in finite):
        raise RuntimeError("Nonfinite recurrence gradient.")
    norms = [float(mx.linalg.norm(grad.astype(mx.float32)).item()) for grad in grads]
    if not all(value > 0 for value in norms):
        raise RuntimeError("Expected a nonzero gradient for each recurrence input.")
    return {
        "sequence_length": length,
        "batch": 1,
        "qk_heads": 16,
        "value_heads": 32,
        "key_value_head_dim": 128,
        "input_dtype": "bfloat16",
        "persistent_recurrence_state_bytes": 32 * 128 * 128 * 4,
        "forward_both_paths_seconds_including_compilation": forward_seconds,
        "forward_max_abs_error": output_error.item(),
        "state_max_abs_error": state_error.item(),
        "forward_backward_seconds_including_compilation": seconds,
        "active_bytes_before_backward": active_before,
        "mlx_peak_allocated_bytes": mx.get_peak_memory(),
        "loss": loss.item(),
        "gradient_norms_q_k_v_a_b": norms,
        "status": "passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("Metal is required; no CPU fallback.")
    mx.set_default_device(mx.gpu)
    results = []
    for length in (16, 64, 128):
        result = probe(length)
        results.append(result)
        print(json.dumps(result), flush=True)
        gc.collect()
        mx.clear_cache()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Single recurrence operator with synthetic inputs; no weights, projections, full layers, optimizer, checkpointing or LLM training",
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")},
        "source_sha256": hashlib.sha256(Path(inspect.getfile(gated_delta)).read_bytes()).hexdigest(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
