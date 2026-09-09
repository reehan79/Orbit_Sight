from __future__ import annotations

"""PART 4 — build all-window Top-20 candidate dataset (Training_sets). Local artifact only."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from orbitsight.evaluation.gt_assignment import nearest_compatible_gt
from orbitsight.features import FEATURE_NAMES, extract_candidate_features
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows
from orbitsight.io import Detection, read_detection_file
from orbitsight.proposals import RawGridProposer, infer_sensor_geometry

PRIOR_MS = 80
TOP_K = 20
SPLIT = Path(r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets")


def sensor_name(sequence: str) -> str:
    u = sequence.upper()
    if u.startswith("DAVIS"):
        return "DAVIS"
    if u.startswith("DVX"):
        return "DVX"
    return "EVK4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=SPLIT)
    parser.add_argument("--out", type=Path, default=Path("artifacts/all_window_candidates.npz"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    features_l, sequences, sensors, starts, ends, ranks, targets, gt_ids = [], [], [], [], [], [], [], []

    for gt_path in sorted(args.split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        sensor = sensor_name(sequence)
        print(f"Building candidates: {sequence}", flush=True)
        arr = np.load(args.split_dir / f"{sequence}_labeled_events.npy", mmap_mode="r")
        ts = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        gts_all = read_detection_file(gt_path)

        for ws in enumerate_challenge_windows(ts):
            we = int(ws) + WINDOW_US
            left = int(np.searchsorted(ts, ws, side="left"))
            right = int(np.searchsorted(ts, we, side="left"))
            prior_left = int(np.searchsorted(ts, int(ws) - PRIOR_MS * 1000, side="left"))
            current = np.asarray(arr[left:right, :4])
            prior = np.asarray(arr[prior_left:left, :4])
            candidates = proposer.propose(current)
            if not candidates:
                continue
            feats = extract_candidate_features(current, prior, candidates, width, height, cell)
            window_gts = [g for g in gts_all if g.start_us < we and g.end_us > int(ws)]
            for rank0, cand in enumerate(candidates):
                gt = nearest_compatible_gt(cand, window_gts, float(cell)) if window_gts else None
                features_l.append(feats[rank0])
                sequences.append(sequence)
                sensors.append(sensor)
                starts.append(int(ws))
                ends.append(we)
                ranks.append(rank0 + 1)
                targets.append(1 if gt is not None else 0)
                if gt is None:
                    gt_ids.append(-1)
                else:
                    gid = -1
                    for i, g in enumerate(gts_all):
                        if g is gt or (
                            g.start_us == gt.start_us
                            and g.end_us == gt.end_us
                            and g.cx == gt.cx
                            and g.cy == gt.cy
                        ):
                            gid = i
                            break
                    gt_ids.append(gid)

    X = np.asarray(features_l, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int8)
    np.savez_compressed(
        args.out,
        features=X,
        sequence=np.asarray(sequences, dtype=object),
        sensor=np.asarray(sensors, dtype=object),
        window_start_us=np.asarray(starts, dtype=np.int64),
        window_end_us=np.asarray(ends, dtype=np.int64),
        candidate_rank=np.asarray(ranks, dtype=np.int16),
        target=y,
        gt_id=np.asarray(gt_ids, dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(
        f"rows={n} positives={n_pos} negatives={n_neg} "
        f"pos_ratio={n_pos / max(n, 1):.6f} disk_mb={size_mb:.2f} out={args.out}",
        flush=True,
    )
    Path("docs/runs/2026-08-30/challenge_metric_baseline").mkdir(parents=True, exist_ok=True)
    Path("docs/runs/2026-08-30/challenge_metric_baseline/all_window_cache_stats.txt").write_text(
        f"rows={n}\npositives={n_pos}\nnegatives={n_neg}\n"
        f"pos_ratio={n_pos / max(n, 1):.6f}\ndisk_mb={size_mb:.2f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
