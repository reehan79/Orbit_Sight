#!/usr/bin/env python3
"""Docker offline smoke: native vs container bit-identity (public V2 preds)."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_pred(path: Path) -> list[tuple]:
    """Load public nine-column TSV; compare geometry+confidence (cols 1..8)."""
    rows = []
    with path.open(encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            rows.append(
                (
                    parts[0],
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                    int(parts[5]),
                    int(parts[6]),
                    int(parts[7]),
                    float(parts[8]),
                )
            )
    return header, rows


def compare_dirs(a: Path, b: Path, atol: float = 1e-12) -> None:
    files_a = sorted(a.glob("*_pred.txt")) + sorted(a.glob("*.txt"))
    # Prefer *_pred.txt; exclude Evaluation note files
    files_a = [p for p in sorted(a.iterdir()) if p.suffix == ".txt" and "Evaluation" not in p.name]
    files_b = {p.name: p for p in b.iterdir() if p.suffix == ".txt" and "Evaluation" not in p.name}
    if not files_a:
        raise SystemExit(f"no predictions in {a}")
    for pa in files_a:
        pb = files_b.get(pa.name)
        if pb is None:
            raise SystemExit(f"missing {pa.name} in {b}")
        ha, ra = load_pred(pa)
        hb, rb = load_pred(pb)
        if ha != hb:
            raise SystemExit(f"header mismatch {ha} vs {hb}")
        if len(ra) != len(rb):
            raise SystemExit(f"{pa.name} length {len(ra)} vs {len(rb)}")
        for i, (xa, xb) in enumerate(zip(ra, rb)):
            if xa[:8] != xb[:8]:
                raise SystemExit(f"{pa.name} row {i} fields differ {xa[:8]} vs {xb[:8]}")
            if abs(xa[8] - xb[8]) > atol:
                raise SystemExit(f"{pa.name} row {i} conf {xa[8]} vs {xb[8]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="sparsesight-ssa:phase1-final-v2")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
        ),
    )
    parser.add_argument("--team", default="SparseSight-SSA")
    parser.add_argument("--day", default="09092026")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        fixture = td_path / "dataset"
        fixture.mkdir()
        seq = "DVX_NOAA6_11416_2025-01-20-19-06-31"
        for suffix in ("_labeled_events.npy", "_bb_windows_40ms.txt"):
            src = args.dataset / f"{seq}{suffix}"
            if not src.exists() and suffix.endswith(".npy"):
                raise SystemExit(f"missing {src}")
            if src.exists():
                shutil.copy2(src, fixture / src.name)

        work_native = td_path / "work_native"
        work_docker_a = td_path / "work_docker_a"
        work_docker_b = td_path / "work_docker_b"
        for w in (work_native, work_docker_a, work_docker_b):
            w.mkdir()

        env = {
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "ORBITSIGHT_DATASET": str(fixture),
            "ORBITSIGHT_WORK": str(work_native),
            "ORBITSIGHT_TEAM_NAME": args.team,
            "ORBITSIGHT_MODEL_DIR": str(ROOT / "models" / "final"),
            "ORBITSIGHT_PRED_FILENAME_MODE": "admin_pred",
        }
        cmd = [
            sys.executable,
            "-m",
            "orbitsight.submission",
            "--dataset",
            str(fixture),
            "--work",
            str(work_native),
            "--model-dir",
            str(ROOT / "models" / "final"),
            "--team",
            args.team,
            "--day",
            args.day,
            "--filename-mode",
            "admin_pred",
        ]
        print("NATIVE...", flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
        native_out = work_native / args.team / args.day
        assert (native_out / f"{seq}_pred.txt").exists()
        assert (native_out / "Evaluation_Metrics.xlsx").exists()

        def docker_run(work: Path) -> Path:
            ds = str(fixture).replace("\\", "/")
            wk = str(work).replace("\\", "/")
            if len(ds) >= 2 and ds[1] == ":":
                ds = f"/{ds[0].lower()}{ds[2:]}"
            if len(wk) >= 2 and wk[1] == ":":
                wk = f"/{wk[0].lower()}{wk[2:]}"
            dcmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-v",
                f"{ds}:/OrbitSight_dataset:ro",
                "-v",
                f"{wk}:/work",
                "-e",
                f"ORBITSIGHT_TEAM_NAME={args.team}",
                "-e",
                "ORBITSIGHT_PRED_FILENAME_MODE=admin_pred",
                args.image,
                "python",
                "-m",
                "orbitsight.submission",
                "--day",
                args.day,
            ]
            print("DOCKER:", " ".join(dcmd), flush=True)
            subprocess.run(dcmd, check=True)
            return work / args.team / args.day

        print("DOCKER run 1...", flush=True)
        out_a = docker_run(work_docker_a)
        print("DOCKER run 2...", flush=True)
        out_b = docker_run(work_docker_b)

        compare_dirs(native_out, out_a)
        compare_dirs(out_a, out_b)
        for o in (out_a, out_b):
            if not (o / "Evaluation_Metrics.xlsx").exists():
                raise SystemExit(f"missing Evaluation_Metrics.xlsx in {o}")
        print("SMOKE OK: native==docker==docker rerun (public V2)", flush=True)


if __name__ == "__main__":
    main()
