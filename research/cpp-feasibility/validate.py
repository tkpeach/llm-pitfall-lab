"""Python is the test oracle/orchestrator; the tested worker is a native C++ process.

No model downloads. The MLX SDK is supplied explicitly; no archived environment imports.
Run from project root with .venv/bin/python research/cpp-feasibility/validate.py.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def disk_preflight():
    policy = tomllib.loads((ROOT / "configs/resources.toml").read_text())
    for name in ("project_limit_bytes", "warning_bytes", "min_free_bytes"):
        if type(policy[name]) is not int or policy[name] < 0:
            raise ValueError(f"invalid resource policy: {name}")
    if policy["warning_bytes"] > policy["project_limit_bytes"]:
        raise ValueError("warning threshold exceeds project limit")
    extras = [Path(p).expanduser().resolve() for p in policy["extra_owned_paths"]]
    if any(p.stat().st_dev != ROOT.stat().st_dev for p in extras):
        raise ValueError("external owned paths require a same-volume budget")
    seen, used, pending = set(), 0, [ROOT, *extras]
    while pending:
        p = pending.pop()
        st = p.lstat()
        identity = st.st_dev, st.st_ino
        if identity in seen:
            continue
        seen.add(identity)
        used += st.st_blocks * 512
        if p.is_dir() and not p.is_symlink():
            pending.extend(p.iterdir())
    free = shutil.disk_usage(ROOT).free
    reserve = 128 * 1024**2  # Tiny fixture/checkpoint/build files, no full weights.
    if used + reserve > policy["project_limit_bytes"] or free - reserve < policy["min_free_bytes"]:
        raise RuntimeError("disk preflight rejected")
    return dict(project_physical_bytes=used, volume_free_bytes=free, reserve_bytes=reserve,
                warning=used >= policy["warning_bytes"], policy=policy)


def main():
    disk = disk_preflight()
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_ops
    if not mx.metal.is_available():
        raise RuntimeError("Metal unavailable; no CPU fallback")
    mx.set_default_device(mx.gpu)
    sdk = Path(sys.prefix) / "lib/python3.12/site-packages/mlx"
    binary = ROOT / "runs/cpp-feasibility/build/cpp_probe"
    metal = sdk / "lib/mlx.metallib"
    run = ROOT / "runs/cpp-feasibility" / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(exist_ok=False)
    report = dict(scope="synthetic C++/MLX feasibility, not full-model acceptance",
                  platform=platform.platform(), python=sys.version,
                  mlx=importlib.metadata.version("mlx"), mlx_lm=importlib.metadata.version("mlx-lm"),
                  sdk=str(sdk), disk_before=disk, checks={}, workers=[])
    report["library_sha256"] = hashlib.sha256((sdk / "lib/libmlx.dylib").read_bytes()).hexdigest()
    report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    report["binary_dependencies"] = subprocess.run(["otool", "-L", str(binary)], capture_output=True, text=True, check=True).stdout
    report["source_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                               for name in ("CMakeLists.txt", "probe.cpp", "validate.py")}

    def command(*args, success=True):
        started = time.monotonic()
        p = subprocess.run([str(binary), str(metal), *map(str, args)],
                           capture_output=True, text=True, timeout=120)
        report["workers"].append(dict(args=list(map(str, args)), returncode=p.returncode,
                                      stdout=p.stdout, stderr=p.stderr,
                                      seconds=time.monotonic() - started))
        if success and p.returncode:
            raise RuntimeError(f"C++ worker failed: {p.stderr}")
        if not success and p.returncode == 0:
            raise RuntimeError("negative test unexpectedly succeeded")
        return p

    def compare(name, actual, expected, atol=2e-6, rtol=2e-4, exact=False):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError(f"shape/dtype mismatch: {name}")
        finite = bool(mx.all(mx.isfinite(actual)).item())
        difference = float(mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()) if actual.size else 0.
        passed = finite and bool((mx.array_equal(actual, expected) if exact else
                                  mx.allclose(actual, expected, atol=atol, rtol=rtol)).item())
        report["checks"][name] = dict(passed=passed, max_abs_error=difference,
                                      shape=list(actual.shape), dtype=str(actual.dtype),
                                      atol=0 if exact else atol, rtol=0 if exact else rtol)
        if not passed:
            raise AssertionError(f"numerical comparison failed: {name}, error={difference}")

    def normal(shape, seed, scale=0.1):
        return mx.random.normal(shape, key=mx.random.key(seed)) * scale

    try:
        f = dict(x=normal((4, 2560), 1), weight=normal((1024, 2560), 2),
                 a=normal((2560, 4), 3, .02), b=normal((4, 1024), 4, .02),
                 target=normal((4, 1024), 5),
                 attn_q=normal((1, 16, 8, 256), 6),
                 attn_k=normal((1, 4, 8, 256), 7), attn_v=normal((1, 4, 8, 256), 8),
                 conv_x=normal((1, 11, 4), 9), conv_w=normal((4, 4, 1), 10),
                 q=normal((1, 8, 16, 128), 11), k=normal((1, 8, 16, 128), 12),
                 v=normal((1, 8, 32, 128), 13),
                 g=mx.sigmoid(normal((1, 8, 32), 14)),
                 beta=mx.sigmoid(normal((1, 8, 32), 15)),
                 state=normal((1, 32, 128, 128), 16, .01))
        mx.save_safetensors(str(run / "inputs.safetensors"), f)
        command("ops", run / "inputs.safetensors", run / "outputs.safetensors")
        out = mx.load(str(run / "outputs.safetensors"))
        qw, scales, biases = mx.quantize(f["weight"], group_size=64, bits=4)
        project = lambda x: mx.quantized_matmul(x, qw, scales, biases, group_size=64, bits=4)
        def loss(x, a, b):
            y = project(x) + ((x @ a) @ b) * .25
            return mx.mean(mx.square(y - f["target"]))
        value, gradients = mx.value_and_grad(loss, argnums=(0, 1, 2))(f["x"], f["a"], f["b"])
        compare("quantized_projection", out["quant_output"], project(f["x"]))
        dense = mx.dequantize(qw, scales, biases, group_size=64, bits=4)
        compare("quantized_vs_dequantized", out["quant_output"], f["x"] @ dense.T)
        compare("lora_loss", out["lora_loss"], value)
        for i, g in enumerate(gradients):
            compare(f"lora_gradient_{i}", out[f"lora_grad_{i}"], g)
            compare(f"gradient_checkpoint_{i}", out[f"checkpoint_grad_{i}"], g)
            if not bool(mx.any(g != 0).item()):
                raise AssertionError("unexpected zero gradient")
        # Independent algebraic gradient through a frozen quantized projection.
        dense_loss = lambda x: mx.mean(mx.square(x @ dense.T + ((x @ f["a"]) @ f["b"]) * .25 - f["target"]))
        compare("gradient_through_frozen_quantized_weight", out["lora_grad_0"], mx.grad(dense_loss)(f["x"]))
        bw = mx.quantize(f["weight"].astype(mx.bfloat16), group_size=64, bits=4)
        def bf_loss(x, a, b):
            y = mx.quantized_matmul(x, *bw, group_size=64, bits=4) + ((x @ a) @ b) * .25
            return mx.mean((y.astype(mx.float32) - f["target"]) ** 2)
        bv, bg = mx.value_and_grad(bf_loss, argnums=(0, 1, 2))(
            *(f[n].astype(mx.bfloat16) for n in ("x", "a", "b")))
        compare("bf16_lora_loss", out["bf16_lora_loss"], bv)
        for i, gradient in enumerate(bg):
            compare(f"bf16_lora_gradient_{i}", out[f"bf16_lora_grad_{i}"], gradient, atol=1e-5, rtol=.01)
            if not bool(mx.any(gradient != 0).item()):
                raise AssertionError("unexpected zero BF16 gradient")
        compare("attention", out["attention"], mx.fast.scaled_dot_product_attention(
            f["attn_q"], f["attn_k"], f["attn_v"], scale=1/16, mask="causal"))
        compare("kv_prefix_decode", out["attention_decode"], out["attention"])
        compare("rms_norm", out["norm"], mx.fast.rms_norm(f["x"], None, 1e-6))
        compare("rope", out["rope"], mx.fast.rope(f["attn_q"], 64, traditional=False, base=10000000, scale=1., offset=0))
        compare("depthwise_conv", out["conv"], mx.conv1d(f["conv_x"], f["conv_w"], groups=4))
        compare("custom_metal_kernel", out["custom_metal"], f["x"] ** 2)
        rp = [f[n] for n in ("q", "k", "v", "g", "beta", "state")]
        y, state = gated_delta_ops(*rp)
        compare("recurrence_output", out["recurrence_output"], y)
        compare("recurrence_state", out["recurrence_state"], state)
        compare("recurrence_chunk_output", out["chunk_output"], y, exact=True)
        compare("recurrence_chunk_state", out["chunk_state"], state, exact=True)
        def rloss(*p):
            y, state = gated_delta_ops(*p)
            return mx.mean(y ** 2) + .01 * mx.mean(state ** 2)
        value, gradients = mx.value_and_grad(rloss, argnums=tuple(range(6)))(*rp)
        compare("recurrence_loss", out["recurrence_loss"], value)
        for i, g in enumerate(gradients):
            compare(f"recurrence_gradient_{i}", out[f"recurrence_grad_{i}"], g, atol=1e-9)
            if not bool(mx.any(g != 0).item()):
                raise AssertionError("unexpected zero recurrent gradient")
        # Fused inference and its backward are separate capabilities, never an automatic fallback.
        command("fused", run / "inputs.safetensors", run / "fused.safetensors")
        fused = mx.load(str(run / "fused.safetensors"))
        compare("fused_recurrence_output", fused["output"], y)
        compare("fused_recurrence_state", fused["state"], state)
        p = subprocess.run([str(binary), str(metal), "fused-grad", str(run / "inputs.safetensors"),
                            str(run / "fused-grad.safetensors")], capture_output=True, text=True, timeout=120)
        report["fused_backward"] = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        if p.returncode == 0:
            fg = mx.load(str(run / "fused-grad.safetensors"))
            for i, g in enumerate(gradients):
                compare(f"fused_recurrence_gradient_{i}", fg[f"grad_{i}"], g, atol=1e-9)
        elif "vjp" not in p.stderr.lower() and "gradient" not in p.stderr.lower():
            raise RuntimeError(f"unexpected fused backward failure: {p.stderr}")
        # Scalar gate directional finite difference, in addition to shared-backend parity.
        eps = .001
        plus, minus = list(rp), list(rp)
        plus[3], minus[3] = rp[3] + eps, rp[3] - eps
        fd = (rloss(*plus) - rloss(*minus)) / (2 * eps)
        compare("recurrence_gate_finite_difference", mx.sum(out["recurrence_grad_3"]), fd, atol=1e-7, rtol=.02)
        report["python_peak_bytes"] = mx.get_peak_memory()
        command("train", run / "continuous.safetensors", 8)
        # Independently check the C++ optimizer against MLX's Python Adam implementation.
        import mlx.optimizers as optim
        command("train", run / "one-step.safetensors", 1)
        a = mx.random.normal((256, 4), scale=.02, key=mx.random.key(12))
        b = mx.zeros((4, 128))
        base = mx.quantize(mx.random.normal((128, 256), scale=.05, key=mx.random.key(7)), group_size=64, bits=4)
        keys = mx.random.split(mx.random.key(42), num=3)
        sample = mx.random.randint(0, 17, (1,), key=keys[1])
        x = mx.random.normal((4, 256), key=keys[2]) + sample.astype(mx.float32) * .01
        def train_loss(p):
            y = mx.quantized_matmul(x, *base, group_size=64, bits=4) + ((x @ p["a"]) @ p["b"]) * .25
            return mx.mean((y - .1) ** 2)
        grad = mx.grad(train_loss)({"a": a, "b": b})
        optimizer = optim.Adam(.01, bias_correction=True)
        expected_params = optimizer.apply_gradients(grad, {"a": a, "b": b})
        one = mx.load(str(run / "one-step.safetensors"))
        for name in ("a", "b"):
            compare(f"adam_parameter_{name}", one[name], expected_params[name])
            for moment in ("m", "v"):
                compare(f"adam_{moment}_{name}", one[moment + name], optimizer.state[name][moment], atol=1e-10)
        command("train", run / "paused.safetensors", 3)
        paused = mx.load(str(run / "paused.safetensors"), return_metadata=True)
        if paused[1]["step"] != "3":
            raise AssertionError("pause progress mismatch")
        command("resume", run / "paused.safetensors", 8)
        continuous = mx.load(str(run / "continuous.safetensors"), return_metadata=True)
        resumed = mx.load(str(run / "paused.safetensors"), return_metadata=True)
        if continuous[1] != resumed[1] or continuous[0].keys() != resumed[0].keys():
            raise AssertionError("resume state schema mismatch")
        for name, value in continuous[0].items():
            compare(f"resume_exact_{name}", resumed[0][name], value, exact=True)
        signal_path = run / "signal.safetensors"
        child = subprocess.Popen([str(binary), str(metal), "signal-test", str(signal_path), "8"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 10
            while not Path(str(signal_path) + ".ready").exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("signal worker did not reach boundary")
                time.sleep(.01)
            requested = time.monotonic()
            child.send_signal(signal.SIGINT)
            stdout, stderr = child.communicate(timeout=10)
            report["signal_pause"] = dict(returncode=child.returncode, pid=child.pid,
                                          seconds=time.monotonic()-requested, stdout=stdout, stderr=stderr,
                                          scope="test rendezvous at update boundary; not full-model latency")
            if child.returncode != 0:
                raise RuntimeError(f"signal pause failed: {stderr}")
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
        _, metadata = mx.load(str(signal_path), return_metadata=True)
        if metadata["step"] != "3":
            raise AssertionError("signal pause failed to preserve progress")
        command("resume", signal_path, 8)
        signal_state = mx.load(str(signal_path))
        for name, value in continuous[0].items():
            compare(f"signal_resume_exact_{name}", signal_state[name], value, exact=True)
        compare("adapter_was_updated", mx.array(bool(mx.any(continuous[0]["b"] != 0).item())), mx.array(True), exact=True)
        command("resume", run / "absent.safetensors", 8, success=False)
        command("train", run / "continuous.safetensors", 8, success=False)
        bad = run / "wrong-identity.safetensors"
        mx.save_safetensors(str(bad), resumed[0], {"identity": "wrong", "step": "3"})
        command("resume", bad, 8, success=False)
        truncated = run / "truncated.safetensors"
        truncated.write_bytes((run / "paused.safetensors").read_bytes()[:32])
        command("resume", truncated, 8, success=False)
        report["status"] = "passed"
    except Exception as e:
        report["status"] = "failed"
        report["error"] = repr(e)
        raise
    finally:
        report["disk_after"] = disk_preflight()
        (run / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(dict(status=report["status"], checks=len(report["checks"]), report=str(run / "report.json"))))


if __name__ == "__main__":
    main()
