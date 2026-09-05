"""Sprint helpers: fold-id parsing and atomic checkpoint writing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def parse_fold_ids(spec: str | None, n_folds: int | None = None) -> list[int] | None:
    """Parse --fold-ids like '0,2,4' or '0-2'. None / empty / 'all' => all folds.

    If n_folds is given, validates range [0, n_folds).
    """
    if spec is None or str(spec).strip() == "" or str(spec).strip().lower() == "all":
        return None
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"invalid fold range: {part}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out:
        return None
    uniq = sorted(set(out))
    if n_folds is not None:
        for fid in uniq:
            if fid < 0 or fid >= n_folds:
                raise ValueError(f"fold id {fid} out of range [0, {n_folds})")
    return uniq


def write_atomic_json(path: Path, payload: dict | list) -> None:
    """Write JSON atomically (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
