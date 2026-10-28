#!/usr/bin/env python3
"""Verify old/new artifact integrity and byte-exact reproduction of model/config/tokenizer payloads."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from qwen35_conversion import digest, load_rules


def verified_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "artifact.json").read_text())
    if manifest["repository"] != "Qwen/Qwen3.5-4B" or manifest["revision"] != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a":
        raise ValueError("artifact source identity mismatch")
    for name, record in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("invalid artifact filename")
        path = directory / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"artifact file integrity mismatch: {directory}/{name}")
    return manifest


def compare(baseline: Path, candidate: Path) -> dict:
    old, new = verified_manifest(baseline), verified_manifest(candidate)
    if old["conversion"] != new["conversion"] or old["source_files"] != new["source_files"]:
        raise ValueError("conversion/source identity changed")
    if old["component"] != new["component"] or old["excluded_components"] != new["excluded_components"]:
        raise ValueError("selected model components changed")
    names = ["model.safetensors", "model.safetensors.index.json", "config.json", *load_rules()["copy_unchanged"]]
    for name in names:
        if old["files"][name] != new["files"].get(name):
            raise ValueError(f"reproduction payload differs: {name}")
    return {"status": "passed", "method": "verified SHA-256 and byte lengths of all listed files before comparison",
            "baseline_manifest_sha256": digest(baseline / "artifact.json"),
            "candidate_manifest_sha256": digest(candidate / "artifact.json"),
            "identical_payload_files": names,
            "additional_provenance_files": sorted(set(new["files"]) - set(old["files"])),
            "scope": "byte-exact model weights, index, config and auxiliary payloads in this pinned environment; provenance manifests intentionally differ"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline.resolve(), args.candidate.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
