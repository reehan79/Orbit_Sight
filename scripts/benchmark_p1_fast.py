"""PART 2 — benchmark OLD P1 reference vs FAST P1 on representative sequences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from orbitsight.evaluation.tii_style import pooled_percentiles
from orbitsight.inference.b_current import SequenceStream
from orbitsight.inference.p1_detector import emit_tii_row, run_p1_window_fast, run_p1_window_reference
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows

SPLIT = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
)
SEQS = [
    "DAVIS_EGS_16908_2024-11-01-19-10-44",
    "DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06",
    "DVX_Filtered_Stars_2025-01-20-19-15-10",
    "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17",
    "2025_12_23_21_12_28_EVK4_mag5.2",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=SPLIT)
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/runs/2026-08-31/challenge_aligned_confidence"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 15)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int8)
    conf = ExtraTreesClassifier(n_estimators=32, max_depth=8, random_state=42, n_jobs=1).fit(X, y)
    Xs = rng.standard_normal((80, 33)).astype(np.float32)
    ys = np.full((80, 2), 0.7, dtype=np.float32)
    size = ExtraTreesRegressor(n_estimators=16, max_depth=6, random_state=42, n_jobs=1).fit(Xs, ys)

    old_samples: list[float] = []
    fast_samples: list[float] = []
    parity_failures = 0
    threshold = 0.3

    for seq in SEQS:
        if not (args.split_dir / f"{seq}_labeled_events.npy").exists():
            continue
        stream = SequenceStream(seq, args.split_dir)
        n = 0
        for ws in enumerate_challenge_windows(stream.timestamps):
            if n >= args.max_windows:
                break
            we = int(ws) + WINDOW_US
            from time import perf_counter_ns

            t0 = perf_counter_ns()
            ref = run_p1_window_reference(stream, int(ws), we, conf, size, threshold=threshold)
            old_samples.append((perf_counter_ns() - t0) / 1e6)
            t1 = perf_counter_ns()
            fast = run_p1_window_fast(stream, int(ws), we, conf, size, threshold=threshold)
            fast_samples.append((perf_counter_ns() - t1) / 1e6)
            if ref is None and fast is None:
                n += 1
                continue
            if ref is None or fast is None or ref.emitted != fast.emitted:
                parity_failures += 1
            elif ref.emitted:
                if emit_tii_row(ref) != emit_tii_row(fast):
                    parity_failures += 1
                elif abs(ref.confidence - fast.confidence) > 1e-12:
                    parity_failures += 1
            n += 1

    rows = []
    for label, samples in [("OLD_P1_reference", old_samples), ("FAST_P1", fast_samples)]:
        p = pooled_percentiles({"ALL": samples})
        rows.append(
            {
                "path": label,
                "n": p["n"],
                "p50_ms": p["p50_ms"],
                "p95_ms": p["p95_ms"],
                "p99_ms": p["p99_ms"],
            }
        )
    out = args.out_dir / "p1_fast_vs_old_latency.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (args.out_dir / "p1_fast_parity.txt").write_text(
        f"parity_failures={parity_failures}\n", encoding="utf-8"
    )
    print(f"parity_failures={parity_failures} csv={out}")


if __name__ == "__main__":
    main()
