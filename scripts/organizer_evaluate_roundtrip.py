#!/usr/bin/env python3
"""Organizer evaluate.py round-trip on a Training_sets sequence copy (not Testing_sets)."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_DataLoader\evaluate.py"
)
TRAIN = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
)
SEQ = "DVX_NOAA6_11416_2025-01-20-19-06-31"


def main() -> int:
    if not EVAL.exists() or not (TRAIN / f"{SEQ}_labeled_events.npy").exists():
        print("SKIP missing organizer materials")
        return 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dataset = td / "dataset"
        work = td / "work"
        dataset.mkdir()
        work.mkdir()
        for suf in ("_labeled_events.npy", "_bb_windows_40ms.txt"):
            shutil.copy2(TRAIN / f"{SEQ}{suf}", dataset / f"{SEQ}{suf}")

        cmd = [
            sys.executable,
            "-m",
            "orbitsight.submission",
            "--dataset",
            str(dataset),
            "--work",
            str(work),
            "--model-dir",
            str(ROOT / "models" / "final"),
            "--team",
            "OrbitSight",
            "--day",
            "08092026",
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        out = work / "OrbitSight" / "08092026"
        pred = out / f"{SEQ}_bb_windows_40ms.txt"
        assert pred.exists(), f"missing {pred}"
        header = pred.read_text(encoding="utf-8").splitlines()[0].split("\t")
        assert header == [
            "window_start_timestamp_us",
            "window_end_timestamp_us",
            "center_x",
            "center_y",
            "width",
            "height",
            "confidence",
        ], header

        excel = td / "Evaluation_Metric.xlsx"
        ev = subprocess.run(
            [
                sys.executable,
                str(EVAL),
                "--gt-dir",
                str(dataset),
                "--pred-dir",
                str(out),
                "--excel-out",
                str(excel),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        text = (ev.stdout or "") + (ev.stderr or "")
        print(text.encode("ascii", errors="replace").decode("ascii"), flush=True)
        if "Missing prediction" in text or "No sequences evaluated" in text:
            print("FAIL organizer evaluate did not consume predictions")
            return 1
        if not excel.exists():
            print("FAIL excel not written")
            return 1
        print("ROUNDTRIP_OK", pred, excel)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
