#!/usr/bin/env python3
"""Compare native/reference execution per chunking mode and compare their measured BF16 drift."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from process import run, require_success

# Established before target execution; retained for corresponding arithmetic paths.
MAX_SCALED_RMSE = 0.01
MAX_LOGIT_ERROR = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resource-binary", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSON containing nonempty text chat messages")
    args = parser.parse_args()
    artifact, root, binary = args.artifact.resolve(), args.root.resolve(), args.binary.resolve()
    checked = run([str(args.resource_binary.resolve()), "resources", "--root", str(root),
                   "--config", str(root / "configs/resources.toml"), "--peak-bytes", str(1024**3)])
    require_success(checked, "reference phase resource preflight")
    import mlx.core as mx
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True, trust_remote_code=False)
    messages = json.loads(args.input.read_text())["messages"]
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("input requires nonempty text chat messages")
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_dict=True)
    ids = encoded["input_ids"]
    if encoded["attention_mask"] != [1] * len(ids):
        raise AssertionError("numerical input unexpectedly includes padding")
    direct = Tokenizer.from_file(str(artifact / "tokenizer.json")).encode(rendered, add_special_tokens=False).ids
    if ids != direct or ids != tokenizer.encode(rendered, add_special_tokens=False):
        raise AssertionError("chat template and tokenizer entry paths disagree")
    if "<|im_start|>assistant\n" not in rendered:
        raise AssertionError("unexpected text conversation template")
    if not rendered.endswith("<think>\n\n</think>\n\n"):
        raise AssertionError("non-thinking generation prefix differs from fixed template")

    metrics, cross_mode, references, workers = {}, {}, {}, {}
    report = {"status": "running", "artifact_manifest_sha256": hashlib.sha256((artifact / "artifact.json").read_bytes()).hexdigest(),
              "input": {"messages": messages, "rendered": rendered, "token_ids": ids},
              "tolerances": {"scaled_rmse": MAX_SCALED_RMSE, "max_logit_absolute_error": MAX_LOGIT_ERROR,
                             "greedy_token_ids": "exact across workers/modes", "shapes_and_dtypes": "exact",
                             "cross_mode": "report raw BF16 drift; compare native drift against reference drift with the same numerical tolerances"},
              "preflight": json.loads(checked.stdout), "reference_workers": references, "native_workers": workers,
              "tensor_metrics": metrics, "cross_mode": cross_mode}

    def save_report() -> None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def fail(message: str) -> None:
        report.update(status="failed", reason=message)
        save_report()
        raise AssertionError(message)

    def difference(actual, expected, scale_source=None) -> dict:
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            fail(f"tensor layout differs: actual={actual.shape}/{actual.dtype}, reference={expected.shape}/{expected.dtype}")
        a, b = actual.astype(mx.float32), expected.astype(mx.float32)
        error = a - b
        if not bool(mx.all(mx.isfinite(error)).item()):
            fail("nonfinite tensor difference")
        scale = b if scale_source is None else scale_source.astype(mx.float32)
        return {"max_absolute_error": mx.max(mx.abs(error)).item(),
                "scaled_rmse": mx.sqrt(mx.mean(error**2)).item() / max(mx.sqrt(mx.mean(scale**2)).item(), 1e-6)}

    def exceeds(metric: dict, name: str) -> bool:
        return metric["scaled_rmse"] > MAX_SCALED_RMSE or (name == "logits" and metric["max_absolute_error"] > MAX_LOGIT_ERROR)

    script = Path(__file__).with_name("reference.py")
    workspace = root / "runs/numerical-validation"
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen35-", dir=workspace) as temp:
        directory = Path(temp)
        input_file = directory / "input.json"
        input_file.write_text(json.dumps({"tokens": ids, "rendered": rendered}, ensure_ascii=False))
        for chunk in (0, 1, 3):
            expected_file = directory / f"reference-{chunk}.safetensors"
            result = run([sys.executable, str(script), "--artifact", str(artifact), "--input", str(input_file),
                          "--output", str(expected_file), "--chunk", str(chunk)], timeout=180)
            if result.returncode != 0:
                fail(f"reference chunk={chunk} failed: {result.stdout}\n{result.stderr}")
            references[str(chunk)] = json.loads(result.stdout.strip().splitlines()[-1])
            output_file = directory / f"native-{chunk}.safetensors"
            result = run([str(binary), str(root), str(artifact), str(input_file), str(output_file), str(chunk), "8"], timeout=180)
            if result.returncode != 0:
                fail(f"native chunk={chunk} failed: {result.stdout}\n{result.stderr}")
            workers[str(chunk)] = json.loads(result.stdout.strip().splitlines()[-1])
            generated = workers[str(chunk)]["generated"]
            expected_offset = len(ids) + len(generated) - int(bool(generated) and generated[-1] in (248044, 248046))
            if workers[str(chunk)]["final_offset"] != expected_offset:
                fail(f"native request progress differs: chunk={chunk}")
            ref, actual = mx.load(str(expected_file)), mx.load(str(output_file))
            if set(actual) != set(ref):
                fail(f"tensor coverage mismatch: chunk={chunk}")
            failures = []
            for name in sorted(ref):
                if name == "generated":
                    if actual[name].shape != ref[name].shape or actual[name].dtype != ref[name].dtype:
                        fail(f"generated token tensor layout differs: chunk={chunk}")
                    if not mx.array_equal(actual[name], ref[name]).item():
                        failures.append(f"greedy generation differs: chunk={chunk}")
                    continue
                metric = difference(actual[name], ref[name])
                metrics[f"chunk{chunk}/{name}"] = metric
                if exceeds(metric, name):
                    failures.append(f"chunk={chunk} {name}: {metric}")
            if chunk:
                native_full = mx.load(str(directory / "native-0.safetensors"))
                reference_full = mx.load(str(directory / "reference-0.safetensors"))
                cross_mode[str(chunk)] = {}
                if not mx.array_equal(actual["generated"], native_full["generated"]).item() or not mx.array_equal(ref["generated"], reference_full["generated"]).item():
                    failures.append(f"cross-mode greedy token IDs differ: chunk={chunk}")
                for name in sorted(ref):
                    if name == "generated":
                        continue
                    native_drift = actual[name].astype(mx.float32) - native_full[name].astype(mx.float32)
                    reference_drift = ref[name].astype(mx.float32) - reference_full[name].astype(mx.float32)
                    metric = difference(native_drift, reference_drift, reference_full[name])
                    cross_mode[str(chunk)][name] = {"native_raw_drift": difference(actual[name], native_full[name]),
                                                   "reference_raw_drift": difference(ref[name], reference_full[name]),
                                                   "drift_difference": metric}
                    if exceeds(metric, name):
                        failures.append(f"cross-mode drift differs from reference: chunk={chunk} {name}: {metric}")
                del native_full, reference_full, native_drift, reference_drift
            if failures:
                fail("; ".join(failures))
            del ref, actual
            mx.clear_cache()
        report.update(status="passed", decoded_generation=tokenizer.decode(references["0"]["generated"]),
                      scope="one short unpadded text request; same-mode native/reference agreement and reference-matched BF16 cross-mode drift; no target training, long-context or bitwise cross-mode equivalence claim")
        save_report()
        print(json.dumps({"status": "passed", "input_tokens": len(ids), "tensor_comparisons": len(metrics), "report": str(args.report)}))


if __name__ == "__main__":
    main()
