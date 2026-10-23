"""D0 repository checks. Static boundaries supplement, not replace, runtime tests."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
from urllib.parse import unquote, urlsplit


def markdown_targets(text: str):
    # Code examples are not navigation. Current docs use inline Markdown links.
    text = re.sub(r"^```[^\n]*\n.*?^```\s*$", "", text, flags=re.M | re.S)
    return re.findall(r"\]\(([^)]+)\)", text)


def heading_ids(text: str):
    ids, counts = set(), {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, flags=re.M):
        base = "".join(c for c in heading.lower() if c.isalnum() or c in "_ -").replace(" ", "-")
        suffix = counts.get(base, 0)
        ids.add(base if not suffix else f"{base}-{suffix}")
        counts[base] = suffix + 1
    return ids


def check_docs(root: Path):
    root = root.resolve()
    index = root / "docs/README.md"
    files = [root / "README.md", root / "AGENTS.md", *sorted((root / "docs").glob("*.md"))]
    for required in (root / "README.md", root / "AGENTS.md", index):
        if not required.is_file():
            raise ValueError(f"missing documentation entry: {required}")
    indexed = set()
    for file in files:
        for target in markdown_targets(file.read_text()):
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue  # Network availability is not a local consistency requirement.
            path = (file.parent / unquote(url.path)).resolve() if url.path else file
            if not path.is_relative_to(root) or not path.exists():
                raise ValueError(f"broken or out-of-repository link: {file.relative_to(root)} -> {target}")
            if url.fragment and path.suffix == ".md" and unquote(url.fragment) not in heading_ids(path.read_text()):
                raise ValueError(f"missing heading: {file.relative_to(root)} -> {target}")
            if file == index:
                indexed.add(path)
    missing = set((root / "docs").glob("*.md")) - indexed - {index}
    if missing:
        raise ValueError(f"formal documents missing from docs/README.md: {sorted(map(str, missing))}")


def check_commands(root: Path, commands: list[dict]):
    if not commands:
        raise ValueError("empty compile database")
    research = (root / "research").resolve()
    for entry in commands:
        directory = Path(entry["directory"])
        source = (directory / entry["file"]).resolve()
        if source.is_relative_to(research):
            raise ValueError(f"research source in formal build: {source}")
        args = entry.get("arguments") or shlex.split(entry["command"])
        standards = [a for a in args if a.startswith("-std=")]
        # CMake uses Clang's c++2b alias for this Apple toolchain. The native
        # build_info header and production build additionally verify the
        # actual __cplusplus level; spelling alone is not the language contract.
        if not standards or standards[-1] not in ("-std=c++23", "-std=c++2b"):
            raise ValueError(f"formal source not built in strict C++23: {source}")
        for i, arg in enumerate(args):
            value = None
            if arg in ("-I", "-isystem", "-iquote") and i + 1 < len(args):
                value = args[i + 1]
            elif arg.startswith("-I") and len(arg) > 2:
                value = arg[2:]
            elif arg.startswith("-isystem") and len(arg) > len("-isystem"):
                value = arg[len("-isystem"):]
            if value and (directory / value).resolve().is_relative_to(research):
                raise ValueError(f"research include path in formal build: {value}")


def check_build(root: Path, build: Path, binary: Path, gpu_binary: Path):
    check_commands(root, json.loads((build / "compile_commands.json").read_text()))
    # Detect explicit research paths in native source. This is deliberately not
    # presented as a proof against arbitrary dynamically constructed paths.
    for file in (root / "src").rglob("*"):
        if file.suffix not in (".cpp", ".hpp", ".h"):
            continue
        if re.search(r"research[/\\]", file.read_text()):
            raise ValueError(f"research path in native source: {file}")
    for executable in (binary, gpu_binary):
        linked = subprocess.check_output(["otool", "-L", str(executable)], text=True)
        if ("libpython" in linked.lower() or "python.framework" in linked.lower()
                or "research/" in linked):
            raise ValueError(f"unexpected native runtime linkage:\n{linked}")
        if executable == gpu_binary and "libmlx.dylib" not in linked:
            raise ValueError("native GPU integration test must link MLX")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("docs", "build"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--gpu-binary", type=Path)
    args = parser.parse_args()
    try:
        if args.kind == "docs":
            check_docs(args.root)
        else:
            if args.build is None or args.binary is None or args.gpu_binary is None:
                parser.error("build checks require --build, --binary and --gpu-binary")
            check_build(args.root.resolve(), args.build.resolve(), args.binary.resolve(), args.gpu_binary.resolve())
        print(f"PASS: {args.kind}")
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"check failed: {error}\n")


if __name__ == "__main__":
    main()
