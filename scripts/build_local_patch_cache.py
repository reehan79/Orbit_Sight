from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from orbitsight.evaluation.gt_assignment import nearest_compatible_gt
from orbitsight.features import FEATURE_NAMES, extract_candidate_features, rasterize_event_patch
from orbitsight.io import Detection, read_detection_file
from orbitsight.proposals import RawGridProposer, infer_sensor_geometry

PRIOR_MS = 80
TOP_K = 20
KEEP_RANKS = (1, 2, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Top-3 local event patch cache (Training_sets only)")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--out", default="artifacts/local_patch_cache.npz")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patches: list[np.ndarray] = []
    features: list[np.ndarray] = []
    sequences: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    ranks: list[int] = []
    cls_targets: list[int] = []
    bbox_targets: list[list[float]] = []

    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        npy_path = split_dir / f"{sequence}_labeled_events.npy"
        arr = np.load(npy_path, mmap_mode="r")
        timestamps = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
        for gt in read_detection_file(gt_path):
            grouped[(gt.start_us, gt.end_us)].append(gt)

        for (start_us, end_us), gts in sorted(grouped.items()):
            left = int(np.searchsorted(timestamps, start_us, side="left"))
            right = int(np.searchsorted(timestamps, end_us, side="left"))
            prior_left = int(np.searchsorted(timestamps, start_us - PRIOR_MS * 1000, side="left"))
            current = np.asarray(arr[left:right, :4])
            prior = np.asarray(arr[prior_left:left, :4])
            candidates = proposer.propose(current)
            if not candidates:
                continue
            feat_mat = extract_candidate_features(current, prior, candidates, width, height, cell)
            for rank in KEEP_RANKS:
                if rank > len(candidates):
                    break
                candidate = candidates[rank - 1]
                patch = rasterize_event_patch(
                    current, prior, candidate.cx, candidate.cy, float(cell), start_us, end_us
                )
                gt = nearest_compatible_gt(candidate, gts, float(cell))
                if gt is None:
                    cls = 0
                    bbox = [math.nan, math.nan, math.nan, math.nan]
                else:
                    cls = 1
                    bbox = [
                        (gt.cx - candidate.cx) / cell,
                        (gt.cy - candidate.cy) / cell,
                        math.log(gt.width / cell),
                        math.log(gt.height / cell),
                    ]
                patches.append(patch)
                features.append(feat_mat[rank - 1].astype(np.float32))
                sequences.append(sequence)
                starts.append(start_us)
                ends.append(end_us)
                ranks.append(rank)
                cls_targets.append(cls)
                bbox_targets.append(bbox)

    patch_arr = np.asarray(patches, dtype=np.float16)
    feat_arr = np.asarray(features, dtype=np.float32)
    bbox_arr = np.asarray(bbox_targets, dtype=np.float32)
    np.savez_compressed(
        out_path,
        patches=patch_arr,
        features=feat_arr,
        sequence=np.asarray(sequences, dtype=object),
        window_start_us=np.asarray(starts, dtype=np.int64),
        window_end_us=np.asarray(ends, dtype=np.int64),
        candidate_rank=np.asarray(ranks, dtype=np.int16),
        cls_target=np.asarray(cls_targets, dtype=np.int8),
        bbox_target=bbox_arr,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    n = len(cls_targets)
    n_pos = int(sum(cls_targets))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"rows={n} positives={n_pos} disk_mb={size_mb:.2f} out={out_path}")


if __name__ == "__main__":
    main()
