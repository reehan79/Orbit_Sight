"""Frozen detector rows → writer → reparse semantic identity (Training fixture optional)."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from orbitsight.evaluation.tii_style import write_tii_prediction_file


def test_writer_preserves_detection_semantics():
    # Frozen known detector rows (internal representation).
    internal = [
        (1_000_000, 1_040_000, 213.2, 187.4, 16.1, 16.9, 0.848222565945547),
        (1_040_000, 1_080_000, 215.0, 185.0, 16.0, 16.0, 0.6330060940344227),
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "DEMO_bb_windows_40ms.txt"
        write_tii_prediction_file(path, internal)
        with path.open(newline="", encoding="utf-8") as f:
            parsed = list(csv.DictReader(f, delimiter="\t"))
        assert len(parsed) == 2
        for src, dst in zip(internal, parsed):
            assert int(dst["window_start_timestamp_us"]) == int(src[0])
            assert int(dst["window_end_timestamp_us"]) == int(src[1])
            assert int(dst["center_x"]) == int(round(src[2]))
            assert int(dst["center_y"]) == int(round(src[3]))
            assert int(dst["width"]) == int(round(src[4]))
            assert int(dst["height"]) == int(round(src[5]))
            assert abs(float(dst["confidence"]) - float(src[6])) <= 1e-12
