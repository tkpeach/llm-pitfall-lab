"""Atomic JSON checkpoints for small mechanism experiments only."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def save_checkpoint(path: Path, identity: dict, state: dict, guard) -> None:
    payload = {"format": 1, "identity": identity, "state": state}
    envelope = {"payload": payload, "sha256": hashlib.sha256(canonical(payload)).hexdigest()}
    data = canonical(envelope)
    # Old committed file remains present; new temporary data adds this much.
    guard.check(len(data) + 65536)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, identity: dict) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    if hashlib.sha256(canonical(payload)).hexdigest() != envelope["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    if payload["format"] != 1 or payload["identity"] != identity:
        raise ValueError("checkpoint format or experiment identity mismatch")
    return payload["state"]
