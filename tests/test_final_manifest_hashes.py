"""Verify models/final/manifest.json SHA256 values match checkout bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "final" / "manifest.json"


def test_final_manifest_file_hashes_match_checkout_bytes():
    assert MANIFEST.exists()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest["files"]
    assert files, "manifest files map empty"
    for name, expected in files.items():
        path = ROOT / "models" / "final" / name
        assert path.exists(), f"missing {path}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, f"hash mismatch for {name}"
