"""Run manifests: every fitted number gets a JSON file naming its inputs and seeds.

The power study is worthless if it is not reproducible (CLAUDE.md), so every
entry point writes one of these next to its output.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import LOGS, RESULTS


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def write_manifest(name: str, payload: dict, inputs: list[Path] | None = None) -> Path:
    record = {
        "name": name,
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_rev": _git_rev(),
        "inputs": {str(p): file_digest(p) for p in (inputs or []) if Path(p).exists()},
        "payload": _jsonable(payload),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def read_manifest(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def append_event(logfile: str, line: str) -> None:
    """Append-only audit line (used by the holdout guard)."""
    path = LOGS / logfile
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with open(path, "a") as fh:
        fh.write(f"{stamp}\t{line}\n")
