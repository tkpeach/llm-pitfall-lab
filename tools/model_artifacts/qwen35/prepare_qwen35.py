#!/usr/bin/env python3
"""Explicit auxiliary preparation of pinned official Qwen3.5-4B and its MLX text artifact."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import urllib.request

REPO = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SMALL_FILES = {"config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
               "merges.txt", "vocab.json", "model.safetensors.index.json", "LICENSE"}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def preflight(root: Path, binary: Path, peak: int) -> dict:
    result = subprocess.run([str(binary), "resources", "--root", str(root), "--config",
                             str(root / "configs/resources.toml"), "--peak-bytes", str(peak)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def fetch(url: str, output: Path, size: int, cancelled: threading.Event | None = None) -> None:
    if size < 64 * 1024**2:
        with urllib.request.urlopen(url, timeout=60) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target, 1024**2)
        return
    # Eight bounded connections; retained chunks can be reused after an interruption.
    chunk_bytes = 32 * 1024**2
    stopping = threading.Event()
    def stopped() -> bool:
        return stopping.is_set() or (cancelled is not None and cancelled.is_set())
    def part(start: int) -> Path:
        end = min(start + chunk_bytes, size) - 1
        file = output.with_name(output.name + f".chunk-{start}")
        for attempt in range(8):
            if stopped():
                raise InterruptedError("download cancelled")
            completed = file.stat().st_size if file.exists() else 0
            if completed == end-start+1:
                return file
            if completed > end-start+1:
                raise ValueError("oversized range temporary file")
            begin = start + completed
            request = urllib.request.Request(url + f"?range_start={begin}", headers={"Range": f"bytes={begin}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=30) as source, file.open("ab") as target:
                    if source.status != 206 or source.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}":
                        raise ValueError("server did not honor the requested byte range")
                    while not stopped() and (chunk := source.read(1024**2)):
                        target.write(chunk)
                if file.stat().st_size == end-start+1:
                    return file
            except (OSError, TimeoutError):
                if attempt == 7:
                    raise
            if stopping.wait(min(attempt + 1, 5)):
                raise InterruptedError("download cancelled")
        raise ValueError(f"incomplete range download after retries: {start}")
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        pending = {pool.submit(part, start): start for start in range(0, size, chunk_bytes)}
        completed_parts = {}
        for future in as_completed(pending):
            completed_parts[pending[future]] = future.result()
        parts = [completed_parts[start] for start in sorted(completed_parts)]
    finally:
        stopping.set()
        pool.shutdown(wait=True, cancel_futures=True)
    with output.open("wb") as target:
        for file in parts:
            with file.open("rb") as source:
                shutil.copyfileobj(source, target, 1024**2)
    # These exact temporary files were created by this transfer; no other cleanup.
    for file in parts:
        file.unlink()


def download(root: Path, binary: Path) -> None:
    target = root / "runs/models/qwen35-4b-official"
    retained = sum(p.stat().st_size for p in target.glob("*") if p.is_file())
    report = preflight(root, binary, max(1024**3, 21 * 1024**3 - retained))
    base = f"https://huggingface.co/{REPO}"
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=30) as source:
        metadata = json.load(source)
    if metadata["sha"] != REVISION:
        raise ValueError("remote revision differs from fixed identity")
    selected = [f for f in metadata["siblings"] if f["rfilename"] in SMALL_FILES or f["rfilename"].endswith(".safetensors")]
    if sum(f["size"] for f in selected) > 10 * 1024**3:
        raise ValueError("official download exceeds the planned 10 GiB source bound")
    target.mkdir(parents=True, exist_ok=True)
    records = {}
    cancelled = threading.Event()
    def one(entry: dict) -> None:
        name = entry["rfilename"]
        if Path(name).name != name:
            raise ValueError("unexpected nested remote file")
        path = target / name
        def verified() -> bool:
            if not path.is_file() or path.stat().st_size != entry["size"]:
                return False
            if "lfs" in entry:
                return digest(path) == entry["lfs"]["sha256"]
            value = hashlib.sha1(f"blob {entry['size']}\0".encode() + path.read_bytes()).hexdigest()
            return value == entry["blobId"]
        if not verified():
            preflight(root, binary, 2 * entry["size"] + 1024**3)
            partial = path.with_name(name + ".partial")
            print(f"Downloading {name} ({entry['size']} bytes)", flush=True)
            fetch(f"{base}/resolve/{REVISION}/{name}", partial, entry["size"], cancelled)
            # Verify before replacing an existing complete source.
            if partial.stat().st_size != entry["size"]:
                raise ValueError(f"download size mismatch: {name}")
            if "lfs" in entry:
                valid = digest(partial) == entry["lfs"]["sha256"]
            else:
                valid = hashlib.sha1(f"blob {entry['size']}\0".encode() + partial.read_bytes()).hexdigest() == entry["blobId"]
            if not valid:
                raise ValueError(f"download hash mismatch: {name}")
            partial.replace(path)
        records[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        pending = [pool.submit(one, entry) for entry in selected]
        for future in as_completed(pending):
            future.result()
    finally:
        cancelled.set()
        pool.shutdown(wait=True, cancel_futures=True)
    (target / "source_manifest.json").write_text(json.dumps({"repository": REPO, "revision": REVISION,
        "files": records, "preflight": report}, indent=2) + "\n")
    print("Official source verified", flush=True)


def convert(root: Path, binary: Path, output: Path | None = None) -> None:
    # Keep the known-good upstream load/sanitize/quantize/save path; add audits.
    from qwen35_conversion import (RULES_FILE, load_rules, converter_identity,
                                   audit_sanitization, audit_quantization)
    rules = load_rules()
    identity = converter_identity(rules)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model, quantize_model, save_model
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    source = root / "runs/models/qwen35-4b-official"
    source_manifest = json.loads((source / "source_manifest.json").read_text())
    if source_manifest["repository"] != REPO or source_manifest["revision"] != REVISION:
        raise ValueError("untrusted conversion source identity")
    for name, record in source_manifest["files"].items():
        if digest(source / name) != record["sha256"]:
            raise ValueError(f"source changed before conversion: {name}")
    target = output.resolve() if output is not None else root / "runs/models/qwen35-4b-text-4bit"
    if target.parent != (root / "runs/models").resolve() or target == source:
        raise ValueError("conversion output must be a distinct direct child of managed runs/models")
    if target.exists():
        raise ValueError("conversion target exists; refusing overwrite")
    preflight(root, binary, 7 * 1024**3)
    print("Loading and sanitizing official language weights", flush=True)
    mx.set_memory_limit(12 * 1024**3)
    audit = {}
    class AuditedModel(Model):
        def sanitize(self, weights):
            result = super().sanitize(weights)
            audit["sanitization"] = audit_sanitization(weights, result, rules)
            return result
    model, config = load_model(source, lazy=True, get_model_classes=lambda config: (AuditedModel, ModelArgs))
    expected = rules["expected_counts"]
    sanitized = audit["sanitization"]
    if sanitized["retained"] != expected["sanitized"] or sanitized["dropped"] != {"vision": expected["source_vision"], "mtp": expected["source_mtp"]}:
        raise ValueError("source component counts differ from the fixed conversion recipe")
    q = rules["quantization"]
    model, config = quantize_model(model, config, q["group_size"], q["bits"], mode=q["mode"])
    audit["quantization"] = audit_quantization(sanitized, dict(tree_flatten(model.parameters())), rules)
    if (audit["quantization"]["quantized_matrices"] != expected["quantized_matrices"] or
            audit["quantization"]["output_tensors"] != expected["output_tensors"]):
        raise ValueError("quantized output counts differ from the recipe")
    config.pop("vision_config", None)
    target.mkdir()
    save_model(target, model, donate_model=True)
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in rules["copy_unchanged"]:
        shutil.copyfile(source / name, target / name)
        if digest(target / name) != source_manifest["files"][name]["sha256"]:
            raise ValueError(f"copied auxiliary file changed: {name}")
    shutil.copyfile(RULES_FILE, target / "conversion-recipe.json")
    (target / "converter-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (target / "conversion-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    # Recheck source files after preparation to prove this process did not rewrite them.
    for name, record in source_manifest["files"].items():
        if digest(source / name) != record["sha256"]:
            raise ValueError(f"official source changed during conversion: {name}")
    records = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(target.iterdir()) if p.is_file()}
    manifest = {"schema_version": 1, "repository": REPO, "revision": REVISION,
                "component": "language_only", "excluded_components": ["vision", "mtp"],
                "conversion": {"mlx": identity["packages"]["mlx"], "mlx_lm": identity["packages"]["mlx-lm"], "bits": 4,
                               "group_size": 64, "mode": "affine", "dtype": "bfloat16"},
                "recipe_id": rules["recipe_id"], "recipe_sha256": identity["recipe_sha256"],
                "source_files": source_manifest["files"], "files": records}
    (target / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"artifact": str(target), "bytes": sum(r['bytes'] for r in records.values()),
                      "peak_memory": mx.get_peak_memory()}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["download", "convert"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new conversion directory under runs/models; never overwrites")
    args = parser.parse_args()
    if args.mode == "download":
        if args.output is not None:
            parser.error("--output applies only to convert")
        download(args.root.resolve(), args.binary.resolve())
    else:
        convert(args.root.resolve(), args.binary.resolve(), args.output)


if __name__ == "__main__":
    main()
