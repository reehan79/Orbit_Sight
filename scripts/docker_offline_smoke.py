#!/usr/bin/env python3
"""Docker offline smoke: native vs container bit-identity (integer boxes)."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_pred(path: Path) -> list[tuple]:
    rows = []
    with path.open(encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            rows.append(
                (
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                    int(parts[5]),
                    float(parts[6]),
                )
            )
    return rows


def compare_dirs(a: Path, b: Path, atol: float = 1e-12) -> None:
    files_a = sorted(a.glob("*_bb_windows_40ms.txt"))
    files_b = {p.name: p for p in b.glob("*_bb_windows_40ms.txt")}
    if not files_a:
        raise SystemExit(f"no predictions in {a}")
    for pa in files_a:
        pb = files_b.get(pa.name)
        if pb is None:
            raise SystemExit(f"missing {pa.name} in {b}")
        ra, rb = load_pred(pa), load_pred(pb)
        if len(ra) != len(rb):
            raise SystemExit(f"{pa.name} length {len(ra)} vs {len(rb)}")
        for i, (xa, xb) in enumerate(zip(ra, rb)):
            if xa[:6] != xb[:6]:
                raise SystemExit(f"{pa.name} row {i} boxes differ {xa[:6]} vs {xb[:6]}")
            if abs(xa[6] - xb[6]) > atol:
                raise SystemExit(f"{pa.name} row {i} conf {xa[6]} vs {xb[6]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="orbitsight-final:latest")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
        ),
    )
    parser.add_argument("--team", default="OrbitSight")
    parser.add_argument("--day", default="08092026")
    args = parser.parse_args()

    # Native run on a tiny subset: copy 1 short sequence into a fixture dataset
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        fixture = td_path / "dataset"
        fixture.mkdir()
        # Prefer a small DAVIS sequence for smoke speed
        seq = "DVX_NOAA6_11416_2025-01-20-19-06-31"
        for suffix in ("_labeled_events.npy",):
            src = args.dataset / f"{seq}{suffix}"
            if not src.exists():
                raise SystemExit(f"missing {src}")
            shutil.copy2(src, fixture / src.name)

        work_native = td_path / "work_native"
        work_docker_a = td_path / "work_docker_a"
        work_docker_b = td_path / "work_docker_b"
        work_native.mkdir()
        work_docker_a.mkdir()
        work_docker_b.mkdir()

        env = {
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "ORBITSIGHT_DATASET": str(fixture),
            "ORBITSIGHT_WORK": str(work_native),
            "ORBITSIGHT_TEAM_NAME": args.team,
            "ORBITSIGHT_MODEL_DIR": str(ROOT / "models" / "final"),
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
        ]
        print("NATIVE...", flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
        native_out = work_native / args.team / args.day

        def docker_run(work: Path) -> Path:
            # Windows Docker Desktop: convert paths
            ds = str(fixture).replace("\\", "/")
            wk = str(work).replace("\\", "/")
            # Prefer // for docker on Windows drive mounts
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
        print("SMOKE OK: native==docker==docker rerun (boxes identical, conf atol 1e-12)", flush=True)


if __name__ == "__main__":
    main()
