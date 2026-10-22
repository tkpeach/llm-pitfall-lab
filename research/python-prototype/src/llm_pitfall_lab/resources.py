"""Conservative preflight, not an OS disk quota or a continuous monitor."""

import os
import shutil
import stat
import tomllib
from pathlib import Path


class ResourceError(RuntimeError):
    pass


def physical_usage(paths: list[Path]) -> int:
    seen = set()
    total = 0
    pending = list(paths)
    while pending:
        path = pending.pop()
        info = path.lstat()  # Missing or inaccessible paths fail closed.
        identity = (info.st_dev, info.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        total += info.st_blocks * 512
        if stat.S_ISDIR(info.st_mode):
            pending.extend(path.iterdir())
        # Do not follow symlinks; owned external data must be explicitly listed.
    return total


class ResourceGuard:
    def __init__(self, root: Path):
        self.root = root.resolve()
        with (self.root / "configs/resources.toml").open("rb") as handle:
            self.policy = tomllib.load(handle)
        for key in ("project_limit_bytes", "warning_bytes", "min_free_bytes"):
            if type(self.policy[key]) is not int or self.policy[key] < 0:
                raise ValueError(f"invalid resource policy: {key}")
        if self.policy["warning_bytes"] > self.policy["project_limit_bytes"]:
            raise ValueError("warning threshold exceeds project limit")

    def check(self, peak_additional_bytes: int) -> dict:
        if type(peak_additional_bytes) is not int or peak_additional_bytes < 0:
            raise ValueError("peak_additional_bytes must be a nonnegative integer")
        extras = [Path(p).expanduser().resolve() for p in self.policy["extra_owned_paths"]]
        if any(p.stat().st_dev != self.root.stat().st_dev for p in extras):
            raise ResourceError("external owned paths on another volume require a separate budget")
        used = physical_usage([self.root, *extras])
        free = shutil.disk_usage(self.root).free
        project_room = self.policy["project_limit_bytes"] - used
        volume_room = free - self.policy["min_free_bytes"]
        report = {"used_bytes": used, "free_bytes": free,
                  "project_room_bytes": project_room, "volume_room_bytes": volume_room,
                  "peak_additional_bytes": peak_additional_bytes,
                  "warning": used >= self.policy["warning_bytes"]}
        if peak_additional_bytes > min(project_room, volume_room):
            raise ResourceError(f"resource preflight rejected: {report}")
        return report
