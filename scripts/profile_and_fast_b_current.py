from __future__ import annotations

"""PART 1–2: profile B_CURRENT and benchmark reference vs fast path."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from orbitsight.inference.b_current import (
    SequenceStream,
    profile_b_current_window,
    run_b_current_fast,
    run_b_current_reference,
)
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows
from orbitsight.models import fit_rankers
from orbitsight.features import FEATURE_NAMES, refine_c1_centroid, extract_local_geometry_features
from orbitsight.proposals import RawGridProposer, infer_sensor_geometry
import math
from collections import defaultdict as dd

PRIOR_MS = 80
TOP_K = 20
RANKER = "M2b_extra_trees"
SPLIT = Path(r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets")

PROFILE_SEQS = [
    "DAVIS_EGS_16908_2024-11-01-19-10-44",
    "DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06",
    "DVX_Filtered_Stars_2025-01-20-19-15-10",
    "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17",
    "2025_12_23_21_12_28_EVK4_mag5.2",
]


def load_table(path: Path):
    sequences, ranks, targets, features, bbox, starts, ends = [], [], [], [], [], [], []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequences.append(row["sequence"])
            starts.append(int(row["window_start_us"]))
            ends.append(int(row["window_end_us"]))
            ranks.append(int(row["candidate_rank"]))
            targets.append(int(row["target"]))
            features.append([float(row[n]) for n in FEATURE_NAMES])
            if row["bbox_log_w_cells"] == "":
                bbox.append([np.nan, np.nan])
            else:
                bbox.append([float(row["bbox_log_w_cells"]), float(row["bbox_log_h_cells"])])
    return {
        "sequence": np.asarray(sequences, object),
        "start": np.asarray(starts, np.int64),
        "end": np.asarray(ends, np.int64),
        "rank": np.asarray(ranks, np.int16),
        "target": np.asarray(targets, np.int8),
        "X": np.asarray(features, np.float32),
        "bbox_log_wh": np.asarray(bbox, np.float32),
    }


def fit_size(table, train_idx, split_dir):
    from sklearn.ensemble import ExtraTreesRegressor

    X_rows, y_rows = [], []
    pos = train_idx[table["target"][train_idx] == 1]
    by_w = dd(list)
    for idx in pos:
        by_w[(str(table["sequence"][idx]), int(table["start"][idx]), int(table["end"][idx]))].append(int(idx))
    for (sequence, start, end), indices in by_w.items():
        arr = np.load(split_dir / f"{sequence}_labeled_events.npy", mmap_mode="r")
        ts = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        left = int(np.searchsorted(ts, start, side="left"))
        right = int(np.searchsorted(ts, end, side="left"))
        current = np.asarray(arr[left:right, :4])
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        cands = proposer.propose(current)
        if not cands:
            continue
        for idx in indices:
            rank = int(table["rank"][idx])
            if rank < 1 or rank > len(cands):
                continue
            cand = cands[rank - 1]
            rcx, rcy = refine_c1_centroid(current, cand.cx, cand.cy, cell)
            local18 = extract_local_geometry_features(current, rcx, rcy, cell, width, height)
            X_rows.append(np.concatenate([table["X"][idx], local18]))
            y_rows.append(table["bbox_log_wh"][idx].tolist())
    model = ExtraTreesRegressor(
        n_estimators=32, max_depth=12, min_samples_leaf=24,
        max_features=None, random_state=42, n_jobs=1,
    )
    model.fit(np.asarray(X_rows, np.float32), np.asarray(y_rows, np.float32))
    return model


def pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=SPLIT)
    parser.add_argument("--table", type=Path, default=Path("artifacts/candidate_table.csv"))
    parser.add_argument("--folds", type=Path, default=Path("sequence_folds.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/runs/2026-08-30/challenge_metric_baseline"))
    parser.add_argument("--min-windows", type=int, default=500)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = load_table(args.table)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    # Use fold 0 train models for profiling (representative)
    train = set(folds[0]["train"])
    train_idx = np.flatnonzero(np.isin(table["sequence"], list(train)))
    print("Fitting ranker+S2 for profiling...", flush=True)
    rankers = fit_rankers(table["X"][train_idx], table["target"][train_idx], table["rank"][train_idx], model_names=[RANKER])
    bundle = rankers[RANKER]
    size_trees = fit_size(table, train_idx, args.split_dir)

    component_rows = []
    latency_old = []
    latency_fast = []
    parity_failures = []

    for sequence in PROFILE_SEQS:
        print(f"Profile {sequence}...", flush=True)
        stream = SequenceStream(sequence, args.split_dir)
        starts = enumerate_challenge_windows(stream.timestamps)
        n = min(len(starts), max(args.min_windows, 500))
        starts = starts[:n]
        comps = defaultdict(list)
        for start in starts:
            end = int(start) + WINDOW_US
            timed = profile_b_current_window(stream, int(start), end, bundle, size_trees)
            comps["event_slice"].append(timed.event_slice_ns / 1e6)
            comps["propose"].append(timed.propose_ns / 1e6)
            comps["features"].append(timed.features_ns / 1e6)
            comps["ranker"].append(timed.ranker_ns / 1e6)
            comps["c1"].append(timed.c1_ns / 1e6)
            comps["c4"].append(timed.c4_ns / 1e6)
            comps["local_geom"].append(timed.local_geom_ns / 1e6)
            comps["s2"].append(timed.s2_ns / 1e6)
            comps["decode"].append(timed.decode_ns / 1e6)
            comps["total"].append(timed.total_ns / 1e6)

            from time import perf_counter_ns
            t0 = perf_counter_ns()
            ref = run_b_current_reference(stream, int(start), end, bundle, size_trees)
            latency_old.append((perf_counter_ns() - t0) / 1e6)
            t0 = perf_counter_ns()
            fast = run_b_current_fast(stream, int(start), end, bundle, size_trees)
            latency_fast.append((perf_counter_ns() - t0) / 1e6)

            if ref.selected is None and fast.selected is None:
                continue
            if ref.selected is None or fast.selected is None:
                parity_failures.append({"sequence": sequence, "start": int(start), "reason": "selected_none_mismatch"})
                continue
            if (ref.selected.grid_x, ref.selected.grid_y) != (fast.selected.grid_x, fast.selected.grid_y):
                parity_failures.append({"sequence": sequence, "start": int(start), "reason": "selected_candidate"})
                continue
            if not np.allclose(ref.features, fast.features, atol=1e-6, rtol=0):
                parity_failures.append({"sequence": sequence, "start": int(start), "reason": "features"})
                continue
            if not np.allclose(ref.scores, fast.scores, atol=1e-6, rtol=0):
                parity_failures.append({"sequence": sequence, "start": int(start), "reason": "scores"})
                continue
            if abs(ref.cx - fast.cx) > 1e-6 or abs(ref.cy - fast.cy) > 1e-6:
                parity_failures.append({"sequence": sequence, "start": int(start), "reason": "centre"})
                continue
            if abs(ref.width - fast.width) > 1e-5 or abs(ref.height - fast.height) > 1e-5:
                parity_failures.append({"sequence": sequence, "start": int(start), "reason": "size"})
                continue

        for name, vals in comps.items():
            component_rows.append(
                {
                    "sequence": sequence,
                    "component": name,
                    "n_windows": len(vals),
                    "p50_ms": pct(vals, 50),
                    "p95_ms": pct(vals, 95),
                    "p99_ms": pct(vals, 99),
                }
            )

    write_csv(args.out_dir / "profile_components.csv", component_rows)
    write_csv(
        args.out_dir / "fast_vs_old_latency.csv",
        [
            {
                "path": "OLD_reference",
                "n": len(latency_old),
                "p50_ms": pct(latency_old, 50),
                "p95_ms": pct(latency_old, 95),
                "p99_ms": pct(latency_old, 99),
            },
            {
                "path": "FAST",
                "n": len(latency_fast),
                "p50_ms": pct(latency_fast, 50),
                "p95_ms": pct(latency_fast, 95),
                "p99_ms": pct(latency_fast, 99),
            },
        ],
    )
    speedup = pct(latency_old, 50) / max(pct(latency_fast, 50), 1e-9)
    (args.out_dir / "fast_parity.txt").write_text(
        f"parity_failures={len(parity_failures)}\nspeedup_p50={speedup:.3f}\n",
        encoding="utf-8",
    )
    if parity_failures:
        write_csv(args.out_dir / "parity_failures.csv", parity_failures)
        print(f"STOP: {len(parity_failures)} parity failures", flush=True)
        raise SystemExit(2)
    print(f"parity_ok speedup_p50={speedup:.3f}", flush=True)


if __name__ == "__main__":
    main()
