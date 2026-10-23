"""Locate/validate the explicitly selected, locked MLX SDK without importing MLX."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import tomllib

PROJECT = Path(__file__).resolve().parents[2]


def sdk_info(selected: Path | None = None) -> dict:
    python_pin = (PROJECT / ".python-version").read_text().strip()
    if python_pin != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise ValueError(f"use the project's Python {python_pin} tools environment")
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text())["project"]
    versions = {}
    for requirement in project["dependencies"]:
        name, sep, version = requirement.partition("==")
        if not sep or metadata.version(name) != version:
            raise ValueError(f"tools environment differs from exact dependency: {requirement}")
        versions[name] = version
    root = Path(metadata.distribution("mlx").locate_file("mlx")).resolve()
    if selected is not None and selected.resolve() != root:
        raise ValueError("LAB_MLX_ROOT differs from the selected Python environment's MLX SDK")
    if root.is_relative_to(PROJECT / "research"):
        raise ValueError("the formal build cannot use an SDK from research/")
    required = ("include/mlx/mlx.h", "lib/libmlx.dylib", "lib/mlx.metallib",
                "share/cmake/MLX/MLXConfig.cmake")
    for relative in required:
        path = root / relative
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"missing SDK file or escaped SDK path: {relative}")
    return {"root": str(root), "version": versions["mlx"], "python": sys.version.split()[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="validate this explicit SDK and print JSON")
    args = parser.parse_args()
    try:
        result = sdk_info(args.root)
        print(json.dumps(result) if args.root is not None else result["root"])
    except (ValueError, OSError, metadata.PackageNotFoundError) as error:
        parser.exit(1, f"SDK error: {error}\n")


if __name__ == "__main__":
    main()
