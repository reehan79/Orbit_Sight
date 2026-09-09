"""Unit tests for sprint fold parsing / atomic checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbitsight.sprint import parse_fold_ids, write_atomic_json, write_atomic_text


def test_parse_fold_ids_all():
    assert parse_fold_ids(None) is None
    assert parse_fold_ids("") is None
    assert parse_fold_ids("all") is None


def test_parse_fold_ids_list_and_range():
    assert parse_fold_ids("0,2,4") == [0, 2, 4]
    assert parse_fold_ids("1-3") == [1, 2, 3]
    assert parse_fold_ids("0,2-4") == [0, 2, 3, 4]


def test_parse_fold_ids_validates_range():
    with pytest.raises(ValueError):
        parse_fold_ids("0,5", n_folds=5)
    with pytest.raises(ValueError):
        parse_fold_ids("3-1")


def test_write_atomic_json(tmp_path: Path):
    path = tmp_path / "ckpt" / "fold0.json"
    write_atomic_json(path, {"fold": 0, "done": True})
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["done"] is True
    assert not list(path.parent.glob("*.tmp"))


def test_write_atomic_text(tmp_path: Path):
    path = tmp_path / "done.txt"
    write_atomic_text(path, "ok\n")
    assert path.read_text(encoding="utf-8") == "ok\n"
