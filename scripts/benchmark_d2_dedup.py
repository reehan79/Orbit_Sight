"""Parity + latency for D2 gate path with geometry reuse."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from orbitsight.evaluation.tii_style import pooled_percentiles
from orbitsight.inference.b_current import SequenceStream
from orbitsight.inference.p1_detector import (
    GATE_FEATURE_DIM,
    benchmark_p1_latency,
    build_gate_features,
    emit_tii_row,
    run_p1_window_fast,
)
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows

SPLIT = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
)
SEQS = {
    "DAVIS": "DAVIS_EGS_16908_2024-11-01-19-10-44",
    "DVX": "DVX_Filtered_Stars_2025-01-20-19-15-10",
    "EVK4": "2025_12_23_21_12_28_EVK4_mag5.2",
}


def _models(rng: np.random.Generator):
    X = rng.standard_normal((200, 15)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int8)
    conf = ExtraTreesClassifier(n_estimators=32, max_depth=8, random_state=42, n_jobs=1).fit(X, y)
    Xs = rng.standard_normal((80, 33)).astype(np.float32)
    ys = np.full((80, 2), 0.7, dtype=np.float32)
    size = ExtraTreesRegressor(n_estimators=16, max_depth=6, random_state=42, n_jobs=1).fit(Xs, ys)
    Xg = rng.standard_normal((120, GATE_FEATURE_DIM)).astype(np.float32)
    yg = (Xg[:, 0] > 0).astype(np.int8)
    scaler = StandardScaler().fit(Xg)
    gate = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    gate.fit(scaler.transform(Xg), yg)
    return conf, size, scaler, gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=SPLIT)
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/runs/2026-09-05/sprint_optimization"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    conf, size, scaler, gate = _models(rng)
    gate_thr = 0.5

    parity_failures = 0
    latency_rows = []

    for sensor, seq in SEQS.items():
        if not (args.split_dir / f"{seq}_labeled_events.npy").exists():
            print(f"MISSING {seq}", flush=True)
            continue
        stream = SequenceStream(seq, args.split_dir)
        n = 0
        for ws in enumerate_challenge_windows(stream.timestamps):
            if n >= args.max_windows:
                break
            we = int(ws) + WINDOW_US
            res = run_p1_window_fast(stream, int(ws), we, conf, size, always_emit=True)
            if res is None:
                n += 1
                continue
            old_gf = build_gate_features(res, stream, size, reuse_geometry=False)
            new_gf = build_gate_features(res, stream, size, reuse_geometry=True)
            if not np.allclose(old_gf, new_gf, atol=1e-12, rtol=0):
                parity_failures += 1
            old_p = float(gate.predict_proba(scaler.transform(old_gf.reshape(1, -1)))[0, 1])
            new_p = float(gate.predict_proba(scaler.transform(new_gf.reshape(1, -1)))[0, 1])
            if abs(old_p - new_p) > 1e-12:
                parity_failures += 1
            if abs(float(res.confidence) - float(res.confidence)) > 1e-12:
                parity_failures += 1
            row_old = emit_tii_row(res)
            row_new = emit_tii_row(res)
            if row_old[:6] != row_new[:6]:
                parity_failures += 1
            emit_old = old_p >= gate_thr
            emit_new = new_p >= gate_thr
            if emit_old != emit_new:
                parity_failures += 1
            n += 1

        old_samples = benchmark_p1_latency(
            stream, conf, size, gate_thr, max_windows=args.max_windows,
            gate_scaler=scaler, gate_clf=gate, gate_threshold=gate_thr, reuse_geometry=False,
        )
        new_samples = benchmark_p1_latency(
            stream, conf, size, gate_thr, max_windows=args.max_windows,
            gate_scaler=scaler, gate_clf=gate, gate_threshold=gate_thr, reuse_geometry=True,
        )
        for label, samples in [("OLD_recompute", old_samples), ("NEW_reuse", new_samples)]:
            p = pooled_percentiles({"ALL": samples})
            latency_rows.append(
                {
                    "sensor": sensor,
                    "path": label,
                    "n": p["n"],
                    "p50_ms": p["p50_ms"],
                    "p95_ms": p["p95_ms"],
                    "p99_ms": p["p99_ms"],
                }
            )
        print(f"{sensor} parity_windows={n}", flush=True)

    with (args.out_dir / "d2_dedup_latency.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(latency_rows[0].keys()))
        w.writeheader()
        w.writerows(latency_rows)
    (args.out_dir / "d2_dedup_parity.txt").write_text(
        f"parity_failures={parity_failures}\n", encoding="utf-8"
    )
    print(f"parity_failures={parity_failures}", flush=True)


if __name__ == "__main__":
    main()
