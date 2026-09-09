"""Unit tests for deterministic dataset split resolution (no real Testing_sets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orbitsight.submission import discover_sequences, resolve_split_dir, sequence_dir_for


def _touch_npy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((2, 4), dtype=np.float64))


def test_prefers_testing_sets_when_both_present(tmp_path: Path):
    _touch_npy(tmp_path / "Training_sets" / "a_labeled_events.npy")
    _touch_npy(tmp_path / "Testing_sets" / "b_labeled_events.npy")
    assert discover_sequences(tmp_path) == ["b"]
    assert resolve_split_dir(tmp_path) == (tmp_path / "Testing_sets").resolve()
    assert sequence_dir_for(tmp_path, "b") == (tmp_path / "Testing_sets").resolve()
    with pytest.raises(FileNotFoundError):
        sequence_dir_for(tmp_path, "a")


def test_direct_root_events(tmp_path: Path):
    _touch_npy(tmp_path / "a_labeled_events.npy")
    assert discover_sequences(tmp_path) == ["a"]
    assert resolve_split_dir(tmp_path) == tmp_path.resolve()


def test_training_sets_only_when_no_testing(tmp_path: Path):
    _touch_npy(tmp_path / "Training_sets" / "a_labeled_events.npy")
    assert discover_sequences(tmp_path) == ["a"]
    assert resolve_split_dir(tmp_path) == (tmp_path / "Training_sets").resolve()


def test_split_mode_both_lists_training_then_testing(tmp_path: Path):
    from orbitsight.submission import list_sequence_jobs

    _touch_npy(tmp_path / "Training_sets" / "a_labeled_events.npy")
    _touch_npy(tmp_path / "Testing_sets" / "b_labeled_events.npy")
    jobs = list_sequence_jobs(tmp_path, split_mode="both")
    assert [j.sequence for j in jobs] == ["a", "b"]
    assert jobs[0].label == "Training"
    assert jobs[1].label == "Testing"


def test_unexpected_nested_layout_fails(tmp_path: Path):
    weird = tmp_path / "Other" / "x_labeled_events.npy"
    _touch_npy(weird)
    with pytest.raises(FileNotFoundError):
        discover_sequences(tmp_path)

