"""Exercise the native CLI from an unrelated working directory."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory() as cwd:
        def invoke(*arguments):
            return subprocess.run([str(binary), *arguments], cwd=cwd, capture_output=True,
                                  text=True, timeout=30)

        help_result = invoke("--help")
        assert help_result.returncode == 0 and "--version" in help_result.stdout, help_result
        assert "check-gpu" not in help_result.stdout, help_result
        version = invoke("--version")
        assert version.returncode == 0 and "C++23, MLX SDK" in version.stdout, version
        for arguments in ((), ("unknown",), ("check-gpu",), ("--version", "unexpected")):
            invalid = invoke(*arguments)
            assert invalid.returncode == 2 and "Usage:" in invalid.stderr, invalid
        print("PASS: native help/version/error contracts")


if __name__ == "__main__":
    main()
