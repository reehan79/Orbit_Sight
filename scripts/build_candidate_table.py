from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from orbitsight.features import FEATURE_NAMES, extract_candidate_features
from orbitsight.io import Detection, read_detection_file
from orbitsight.proposals import Candidate, RawGridProposer, infer_sensor_geometry


def _compatible(candidate: Candidate, gt: Detection, margin: float) -> bool:
    return (
        abs(candidate.cx - gt.cx) <= gt.width / 2.0 + margin
        and abs(candidate.cy - gt.cy) <= gt.height / 2.0 + margin
    )


def _best_gt(candidate: Candidate, gts: list[Detection], margin: float) -> Detection | None:
    matches = [gt for gt in gts if _compatible(candidate, gt, margin)]
    if not matches:
        return None
    return min(matches, key=lambda gt: (candidate.cx - gt.cx) ** 2 + (candidate.cy - gt.cy) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the first candidate-ranking table from GT-positive training windows. "
            "Features are label-free; GT is used only for candidate labels/regression targets."
        )
    )
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--out", default="artifacts/candidate_table.csv")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--prior-ms", type=int, default=80)
    parser.add_argument("--max-sequences", type=int, default=0, help="0 means all sequences")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sequence",
        "sensor",
        "window_start_us",
        "window_end_us",
        "candidate_rank",
        "target",
        "bbox_dx_cells",
        "bbox_dy_cells",
        "bbox_log_w_cells",
        "bbox_log_h_cells",
        *FEATURE_NAMES,
    ]

    gt_files = sorted(split_dir.glob("*_bb_windows_40ms.txt"))
    if args.max_sequences > 0:
        gt_files = gt_files[: args.max_sequences]

    total_rows = 0
    positive_rows = 0
    target_boxes = 0
    target_boxes_preserved = 0

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for gt_path in gt_files:
            sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
            npy_path = split_dir / f"{sequence}_labeled_events.npy"
            arr = np.load(npy_path, mmap_mode="r")
            timestamps = arr[:, 3]
            width, height, cell = infer_sensor_geometry(sequence)
            sensor = "DAVIS" if sequence.upper().startswith("DAVIS") else ("DVX" if sequence.upper().startswith("DVX") else "EVK4")
            proposer = RawGridProposer(width, height, cell, top_k=args.top_k)

            grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
            for gt in read_detection_file(gt_path):
                grouped[(gt.start_us, gt.end_us)].append(gt)

            sequence_rows = 0
            sequence_positive = 0
            sequence_targets = sum(len(gts) for gts in grouped.values())
            sequence_preserved = 0

            for (start_us, end_us), gts in sorted(grouped.items()):
                left = int(np.searchsorted(timestamps, start_us, side="left"))
                right = int(np.searchsorted(timestamps, end_us, side="left"))
                prior_start = start_us - args.prior_ms * 1000
                prior_left = int(np.searchsorted(timestamps, prior_start, side="left"))

                current = np.asarray(arr[left:right, :4])
                prior = np.asarray(arr[prior_left:left, :4])
                candidates = proposer.propose(current)
                features = extract_candidate_features(current, prior, candidates, width, height, cell)

                for gt in gts:
                    if any(_compatible(c, gt, float(cell)) for c in candidates):
                        sequence_preserved += 1

                for rank0, (candidate, feature_row) in enumerate(zip(candidates, features), start=1):
                    gt = _best_gt(candidate, gts, float(cell))
                    target = int(gt is not None)
                    if gt is None:
                        dx = dy = log_w = log_h = ""
                    else:
                        dx = (gt.cx - candidate.cx) / cell
                        dy = (gt.cy - candidate.cy) / cell
                        log_w = float(np.log(max(gt.width, 1.0) / cell))
                        log_h = float(np.log(max(gt.height, 1.0) / cell))

                    row: dict[str, object] = {
                        "sequence": sequence,
                        "sensor": sensor,
                        "window_start_us": start_us,
                        "window_end_us": end_us,
                        "candidate_rank": rank0,
                        "target": target,
                        "bbox_dx_cells": dx,
                        "bbox_dy_cells": dy,
                        "bbox_log_w_cells": log_w,
                        "bbox_log_h_cells": log_h,
                    }
                    for name, value in zip(FEATURE_NAMES, feature_row):
                        row[name] = float(value)
                    writer.writerow(row)
                    sequence_rows += 1
                    sequence_positive += target

            total_rows += sequence_rows
            positive_rows += sequence_positive
            target_boxes += sequence_targets
            target_boxes_preserved += sequence_preserved
            recall = sequence_preserved / sequence_targets if sequence_targets else float("nan")
            print(
                f"{sequence:68s} rows={sequence_rows:6d} positives={sequence_positive:5d} "
                f"proposal_recall={100*recall:6.2f}%"
            )

    print("\n=== CANDIDATE TABLE COMPLETE ===")
    print(f"rows={total_rows}")
    print(f"positive_rows={positive_rows}")
    print(f"target_boxes={target_boxes}")
    print(f"target_boxes_preserved={target_boxes_preserved}")
    print(f"top{args.top_k}_proposal_recall={100*target_boxes_preserved/max(target_boxes,1):.3f}%")
    print(f"out={out_path}")
    print("IMPORTANT: this first table contains GT-positive windows only. It is for candidate ranking/refinement, not final empty-window false-positive calibration.")


if __name__ == "__main__":
    main()
