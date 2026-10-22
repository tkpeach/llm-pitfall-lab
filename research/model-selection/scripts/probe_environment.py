"""F0: verify GPU arithmetic and gradients without downloading model weights."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm.models import granite


def probe() -> dict:
    if not mx.metal.is_available():
        raise RuntimeError("Metal is unavailable; CPU fallback is not a passing result.")
    mx.set_default_device(mx.gpu)
    mx.reset_peak_memory()

    # Use tiny, exactly representable values with a hand-checkable product.
    left = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    right = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
    product = left @ right
    mx.eval(product)
    expected_product = [[19.0, 22.0], [43.0, 50.0]]
    if product.tolist() != expected_product:
        raise RuntimeError("GPU matrix multiplication differs from the reference.")

    def loss(weights):
        return mx.mean(mx.square(weights))

    before, gradient = mx.value_and_grad(loss)(left)
    updated = left - 0.1 * gradient
    after = loss(updated)
    expected_gradient = 2 * left / left.size
    gradient_error = mx.max(mx.abs(gradient - expected_gradient))
    mx.eval(before, gradient, updated, after, gradient_error)
    mx.synchronize()
    if gradient_error.item() > 1e-6 or after.item() >= before.item():
        raise RuntimeError("Gradient or update does not match the expected behavior.")

    # Parse real model configs, but never instantiate a full model here.
    root = Path(__file__).resolve().parents[1]
    candidates = json.loads((root / "evidence/model-candidates.json").read_text())
    parsed = []
    for candidate in candidates:
        args = granite.ModelArgs.from_dict(candidate["config"])
        parsed.append({"repo": candidate["repo"], "layers": args.num_hidden_layers})
    module_path = Path(inspect.getfile(granite))
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "GPU arithmetic, gradient/update, and config parsing only; no LLM run",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("mlx", "mlx-metal", "mlx-lm", "transformers")},
        "device": str(mx.default_device()),
        "device_info": mx.device_info(),
        "matrix_product": product.tolist(),
        "gradient": gradient.tolist(),
        "gradient_max_abs_error": gradient_error.item(),
        "loss_before": before.item(),
        "loss_after_one_update": after.item(),
        "mlx_peak_allocated_bytes": mx.get_peak_memory(),
        "granite_module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "parsed_configs": parsed,
        "status": "passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    report = probe()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
