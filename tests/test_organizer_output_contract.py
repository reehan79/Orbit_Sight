"""Organizer evaluate.py contract probes (no real Testing_sets)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

EVAL = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_DataLoader\evaluate.py"
)

# Organizer evaluate.py prints Unicode arrows; force UTF-8 on Windows consoles (cp1252).
_EVAL_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

PRED_HEADER = (
    "window_start_timestamp_us\twindow_end_timestamp_us\t"
    "center_x\tcenter_y\twidth\theight\tconfidence\n"
)
GT_HEADER = (
    "window_start_timestamp_us\twindow_end_timestamp_us\t"
    "center_x\tcenter_y\twidth\theight\n"
)
ROW = "0\t40000\t10\t10\t4\t4\t0.9\n"
GT_ROW = "0\t40000\t10\t10\t4\t4\n"


@pytest.mark.skipif(not EVAL.exists(), reason="organizer evaluate.py not present")
def test_organizer_evaluate_accepts_bb_windows_filename():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        gt, pred = td / "gt", td / "pred"
        gt.mkdir()
        pred.mkdir()
        seq = "SYNTH_SEQ"
        (gt / f"{seq}_bb_windows_40ms.txt").write_text(GT_HEADER + GT_ROW, encoding="utf-8")
        (pred / f"{seq}_bb_windows_40ms.txt").write_text(PRED_HEADER + ROW, encoding="utf-8")
        excel = td / "Evaluation_Metric.xlsx"
        r = subprocess.run(
            [
                sys.executable,
                str(EVAL),
                "--gt-dir",
                str(gt),
                "--pred-dir",
                str(pred),
                "--excel-out",
                str(excel),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_EVAL_ENV,
        )
        text = (r.stdout or "") + (r.stderr or "")
        assert r.returncode == 0
        assert "Missing prediction" not in text
        assert "No sequences evaluated" not in text
        assert excel.exists()


@pytest.mark.skipif(not EVAL.exists(), reason="organizer evaluate.py not present")
def test_organizer_evaluate_rejects_pred_txt_filename():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        gt, pred = td / "gt", td / "pred"
        gt.mkdir()
        pred.mkdir()
        seq = "SYNTH_SEQ"
        (gt / f"{seq}_bb_windows_40ms.txt").write_text(GT_HEADER + GT_ROW, encoding="utf-8")
        (pred / f"{seq}_pred.txt").write_text(PRED_HEADER + ROW, encoding="utf-8")
        excel = td / "out.xlsx"
        r = subprocess.run(
            [
                sys.executable,
                str(EVAL),
                "--gt-dir",
                str(gt),
                "--pred-dir",
                str(pred),
                "--excel-out",
                str(excel),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_EVAL_ENV,
        )
        text = (r.stdout or "") + (r.stderr or "")
        assert "Missing prediction for: SYNTH_SEQ_bb_windows_40ms.txt" in text
        assert "No sequences evaluated" in text
        assert not excel.exists()


def test_internal_tii_writer_still_matches_evaluate_load_pred():
    """Internal evaluate.py helper remains available; production uses public V2 writer."""
    from orbitsight.evaluation.tii_style import write_tii_prediction_file
    import csv

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "SEQ_bb_windows_40ms.txt"
        rows = [(100, 40100, 11.4, 12.6, 5.2, 6.8, 0.848222565945547)]
        write_tii_prediction_file(path, rows)
        text = path.read_text(encoding="utf-8")
        header = text.splitlines()[0].split("\t")
        assert header == [
            "window_start_timestamp_us",
            "window_end_timestamp_us",
            "center_x",
            "center_y",
            "width",
            "height",
            "confidence",
        ]
        with path.open(newline="", encoding="utf-8") as f:
            parsed = list(csv.DictReader(f, delimiter="\t"))
        assert len(parsed) == 1
        p = parsed[0]
        assert int(p["center_x"]) == 11
        assert abs(float(p["confidence"]) - 0.848222565945547) <= 1e-12



def test_work_output_path_contract():
    from orbitsight.submission import output_dir
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = output_dir(Path(td), team_name="SparseSight-SSA", day="08092026")
        assert out == Path(td) / "SparseSight-SSA" / "08092026"
        assert out.is_dir()
