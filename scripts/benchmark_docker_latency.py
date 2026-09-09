#!/usr/bin/env python3
"""Benchmark Docker complete-path latency by sensor (production image)."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

LATENCY_SEQS = [
    ("DAVIS", "DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06"),
    ("DVX", "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17"),
    ("DVX", "DVX_Filtered_Stars_2025-01-20-19-15-10"),
    ("EVK4", "2025_12_23_21_12_28_EVK4_mag5.2"),
]


def win_path_to_docker(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return f"/{s[0].lower()}{s[2:]}"
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="orbitsight-final:latest")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path(
            r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
        ),
    )
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "runs" / "2026-09-08" / "docker_latency" / "latency_by_sensor.csv",
    )
    args = parser.parse_args()

    # Run a small Python latency harness inside the container that imports the
    # same frozen path used by submission.
    harness = r'''
import json, sys
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from orbitsight.submission import load_final_bundle, score_gate
from orbitsight.inference.b_current import SequenceStream
from orbitsight.inference.p1_detector import build_gate_features, emit_tii_row, run_p1_window_fast
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows

bundle = load_final_bundle(Path("/models/final"))
seq = sys.argv[1]
max_w = int(sys.argv[2])
stream = SequenceStream(seq, Path("/OrbitSight_dataset"))
samples = []
n = 0
for ws in enumerate_challenge_windows(stream.timestamps):
    if n >= max_w:
        break
    we = int(ws) + WINDOW_US
    t0 = perf_counter_ns()
    res = run_p1_window_fast(stream, int(ws), we, bundle.conf_model, bundle.size_trees, always_emit=True)
    if res is not None:
        gf = build_gate_features(res, stream, bundle.size_trees, reuse_geometry=True)
        gate_p = score_gate(bundle, gf)
        if gate_p >= bundle.threshold:
            emit_tii_row(res, confidence=gate_p)
    samples.append((perf_counter_ns() - t0) / 1e6)
    n += 1
print(json.dumps({"sequence": seq, "n": len(samples), "samples_ms": samples}))
'''

    rows = []
    all_samples: list[float] = []
    by_sensor: dict[str, list[float]] = {}
    ds = win_path_to_docker(args.split_dir)
    for sensor, seq in LATENCY_SEQS:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{ds}:/OrbitSight_dataset:ro",
            args.image,
            "python",
            "-c",
            harness,
            seq,
            str(args.max_windows),
        ]
        print(f"latency {sensor} {seq}...", flush=True)
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # last JSON line
        payload = None
        for line in proc.stdout.splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{"):
                payload = json.loads(line)
                break
        if payload is None:
            raise SystemExit(f"no JSON from docker for {seq}\n{proc.stdout}\n{proc.stderr}")
        samples = [float(x) for x in payload["samples_ms"]]
        by_sensor.setdefault(sensor, []).extend(samples)
        all_samples.extend(samples)
        rows.append(
            {
                "sensor": sensor,
                "sequence": seq,
                "n": len(samples),
                "p50_ms": float(np.percentile(samples, 50)),
                "p95_ms": float(np.percentile(samples, 95)),
                "p99_ms": float(np.percentile(samples, 99)),
            }
        )

    for sensor, samples in sorted(by_sensor.items()):
        rows.append(
            {
                "sensor": sensor,
                "sequence": "ALL",
                "n": len(samples),
                "p50_ms": float(np.percentile(samples, 50)),
                "p95_ms": float(np.percentile(samples, 95)),
                "p99_ms": float(np.percentile(samples, 99)),
            }
        )
    rows.append(
        {
            "sensor": "ALL",
            "sequence": "ALL",
            "n": len(all_samples),
            "p50_ms": float(np.percentile(all_samples, 50)),
            "p95_ms": float(np.percentile(all_samples, 95)),
            "p99_ms": float(np.percentile(all_samples, 99)),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {r["sensor"] + "/" + r["sequence"]: r for r in rows if r["sequence"] == "ALL"}
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    overall = next(r for r in rows if r["sensor"] == "ALL")
    print(f"OVERALL_P95={overall['p95_ms']:.3f} criterion_le_40={overall['p95_ms'] <= 40.0}", flush=True)


if __name__ == "__main__":
    main()
