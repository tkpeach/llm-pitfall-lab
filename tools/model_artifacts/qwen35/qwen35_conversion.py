"""Audit the pinned upstream conversion against explicit rules; never replace its sanitizer/quantizer."""
from __future__ import annotations
from collections import Counter
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import platform

RULES_FILE = Path(__file__).with_name("qwen35_conversion_rules.json")


def digest(file: Path) -> str:
    with file.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_rules() -> dict:
    rules = json.loads(RULES_FILE.read_text())
    if rules["schema_version"] != 1:
        raise ValueError("unsupported conversion recipe version")
    return rules


def converter_identity(rules: dict) -> dict:
    packages = {name: version(name) for name in rules["packages"]}
    if packages != rules["packages"]:
        raise ValueError("conversion package versions differ from the recipe")
    verified = []
    for record in rules["critical_files"]:
        file = Path(distribution(record["package"]).locate_file(record["path"]))
        actual = digest(file)
        if actual != record["sha256"]:
            raise ValueError(f"converter file differs from recipe: {record['path']}")
        verified.append(dict(record))
    return {"packages": packages, "critical_files": verified, "python": platform.python_version(),
            "platform": platform.platform(), "recipe_id": rules["recipe_id"],
            "recipe_sha256": digest(RULES_FILE),
            "project_sources": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("prepare_qwen35.py"))}}


def describe(tensor) -> dict:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def sanitization_plan(raw: dict, rules: dict) -> list[dict]:
    rule = rules["sanitization"]
    original_conv = any(name.startswith(rule["input_prefix"]) and name.endswith(rule["conv_suffix"])
                        and tensor.shape[-1] != 1 for name, tensor in raw.items())
    records = []
    for name, tensor in sorted(raw.items()):
        entry = {"source": name, "input": describe(tensor)}
        dropped = next((reason for prefix, reason in rule["drop_prefixes"].items() if name.startswith(prefix)), None)
        if dropped:
            entry.update(action="drop", reason=dropped)
        elif name.startswith(rule["input_prefix"]):
            target = rule["output_prefix"] + name[len(rule["input_prefix"]):]
            operations = ["rename_prefix"]
            if name.endswith(rule["conv_suffix"]) and tensor.shape[-1] != 1:
                operations.append("moveaxis_2_to_1")
            if original_conv and tensor.ndim == 1 and any(target.endswith(suffix) for suffix in rule["offset_norm_suffixes"]):
                operations.append("norm_weight_plus_one")
            entry.update(action="retain", target=target, operations=operations)
        else:
            raise ValueError(f"source tensor outside the fixed conversion recipe: {name}")
        records.append(entry)
    return records


def audit_sanitization(raw: dict, sanitized: dict, rules: dict) -> dict:
    import mlx.core as mx
    records = sanitization_plan(raw, rules)
    targets = {entry["target"] for entry in records if entry["action"] == "retain"}
    if set(sanitized) != targets:
        raise ValueError("upstream sanitizer output tensor names differ from declared rules")
    operations = Counter()
    dropped = Counter()
    for entry in records:
        if entry["action"] == "drop":
            dropped[entry["reason"]] += 1
            continue
        original = raw[entry["source"]]
        expected = original
        for operation in entry["operations"]:
            operations[operation] += 1
            if operation == "moveaxis_2_to_1":
                expected = mx.moveaxis(expected, 2, 1)
            elif operation == "norm_weight_plus_one":
                expected = expected + rules["sanitization"]["norm_offset"]
        actual = sanitized[entry["target"]]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"upstream conversion layout differs from rules: {entry['source']}")
        if len(entry["operations"]) == 1:
            # Renames must preserve the actual array; avoid materializing large matrices.
            if actual is not original:
                raise ValueError(f"unlisted modification to retained tensor: {entry['source']}")
        elif not bool(mx.array_equal(actual, expected).item()):
            raise ValueError(f"upstream conversion values differ from rules: {entry['source']}")
        entry["output"] = describe(actual)
    return {"records": records, "retained": len(targets), "dropped": dict(dropped), "operations": dict(operations)}


def audit_quantization(sanitization: dict, quantized: dict, rules: dict) -> dict:
    import mlx.core as mx
    expected_names = set()
    records = []
    group, bits = rules["quantization"]["group_size"], rules["quantization"]["bits"]
    for entry in sanitization["records"]:
        if entry["action"] != "retain":
            continue
        name, info = entry["target"], entry["output"]
        shape = info["shape"]
        if name.endswith(".weight") and len(shape) == 2 and shape[1] % group == 0:
            prefix = name[:-len(".weight")]
            names = [name, prefix + ".scales", prefix + ".biases"]
            shapes = [[shape[0], shape[1] * bits // 32], [shape[0], shape[1] // group], [shape[0], shape[1] // group]]
            for i, (output_name, expected_shape) in enumerate(zip(names, shapes)):
                if output_name not in quantized or list(quantized[output_name].shape) != expected_shape:
                    raise ValueError(f"quantized tensor shape/name differs from recipe: {output_name}")
                dtype = str(mx.uint32) if i == 0 else info["dtype"]
                if str(quantized[output_name].dtype) != dtype:
                    raise ValueError(f"quantized tensor dtype differs from recipe: {output_name}")
            expected_names.update(names)
            records.append({"input": name, "input_layout": info,
                            "outputs": {n: describe(quantized[n]) for n in names}})
        else:
            if name not in quantized or describe(quantized[name]) != info:
                raise ValueError(f"non-quantized tensor layout changed: {name}")
            expected_names.add(name)
    if set(quantized) != expected_names:
        raise ValueError("quantizer produced unexpected tensors")
    return {"quantized_matrices": len(records), "output_tensors": len(quantized), "records": records}
