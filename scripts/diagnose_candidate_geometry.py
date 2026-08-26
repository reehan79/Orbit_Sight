from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from orbitsight.features import FEATURE_NAMES
from orbitsight.proposals import infer_sensor_geometry


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1 = a[0] - a[2] / 2.0, a[1] - a[3] / 2.0
    ax2, ay2 = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
    bx1, by1 = b[0] - b[2] / 2.0, b[1] - b[3] / 2.0
    bx2, by2 = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def sensor_name(sequence: str) -> str:
    u = sequence.upper()
    if u.startswith("DAVIS"):
        return "DAVIS"
    if u.startswith("DVX"):
        return "DVX"
    return "EVK4"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle diagnostic for the candidate-localization bottleneck. "
            "This is not a deployable metric: GT is deliberately used to separate "
            "candidate-center error from box-size error."
        )
    )
    parser.add_argument("--table", required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    with Path(args.table).open("r", encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            if int(r["target"]) != 1:
                continue
            sequence = r["sequence"]
            width, height, cell = infer_sensor_geometry(sequence)
            cx = float(r["cx_normalized"]) * width
            cy = float(r["cy_normalized"]) * height
            dx = float(r["bbox_dx_cells"])
            dy = float(r["bbox_dy_cells"])
            bw = math.exp(float(r["bbox_log_w_cells"])) * cell
            bh = math.exp(float(r["bbox_log_h_cells"])) * cell
            gt_cx = cx + dx * cell
            gt_cy = cy + dy * cell
            rows.append(
                {
                    "sequence": sequence,
                    "sensor": sensor_name(sequence),
                    "start": int(r["window_start_us"]),
                    "end": int(r["window_end_us"]),
                    "candidate_cx": cx,
                    "candidate_cy": cy,
                    "gt_cx": gt_cx,
                    "gt_cy": gt_cy,
                    "gt_w": bw,
                    "gt_h": bh,
                }
            )

    if not rows:
        raise SystemExit("No positive candidate rows found")

    # Use one unique GT box per window/object to estimate sensor-conditioned median size.
    unique_boxes: dict[tuple[object, ...], tuple[str, float, float]] = {}
    for r in rows:
        key = (
            r["sequence"],
            r["start"],
            r["end"],
            round(float(r["gt_cx"]), 4),
            round(float(r["gt_cy"]), 4),
            round(float(r["gt_w"]), 4),
            round(float(r["gt_h"]), 4),
        )
        unique_boxes[key] = (str(r["sensor"]), float(r["gt_w"]), float(r["gt_h"]))

    median_size: dict[str, tuple[float, float]] = {}
    for sensor in ("DAVIS", "DVX", "EVK4"):
        vals = [(w, h) for s, w, h in unique_boxes.values() if s == sensor]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            median_size[sensor] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))

    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        grouped[(str(r["sequence"]), int(r["start"]), int(r["end"]))].append(r)

    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for group in grouped.values():
        # Each positive row may correspond to one of multiple GT objects. Evaluate the
        # best compatible proposal as an oracle upper bound on grid-center localization.
        for r in group:
            sensor = str(r["sensor"])
            gt = (float(r["gt_cx"]), float(r["gt_cy"]), float(r["gt_w"]), float(r["gt_h"]))
            candidate_oracle_size = (
                float(r["candidate_cx"]),
                float(r["candidate_cy"]),
                float(r["gt_w"]),
                float(r["gt_h"]),
            )
            mw, mh = median_size[sensor]
            candidate_median_size = (
                float(r["candidate_cx"]),
                float(r["candidate_cy"]),
                mw,
                mh,
            )
            oracle_center_median_size = (
                float(r["gt_cx"]),
                float(r["gt_cy"]),
                mw,
                mh,
            )
            stats[sensor]["candidate_center_oracle_size"].append(iou(candidate_oracle_size, gt))
            stats[sensor]["candidate_center_median_size"].append(iou(candidate_median_size, gt))
            stats[sensor]["oracle_center_median_size"].append(iou(oracle_center_median_size, gt))

    print("=== CANDIDATE GEOMETRY DIAGNOSTIC (GT-ORACLE; NOT DEPLOYABLE) ===")
    print(f"positive_candidate_rows={len(rows)} unique_gt_boxes={len(unique_boxes)}")
    for sensor in ("DAVIS", "DVX", "EVK4"):
        if sensor not in stats:
            continue
        mw, mh = median_size[sensor]
        print(f"\n{sensor}: median_gt_size={mw:.2f}x{mh:.2f}")
        for name in (
            "candidate_center_oracle_size",
            "oracle_center_median_size",
            "candidate_center_median_size",
        ):
            arr = np.asarray(stats[sensor][name], dtype=np.float64)
            print(
                f"  {name:30s} meanIoU={np.mean(arr):.4f} "
                f"IoU50={100*np.mean(arr >= 0.5):6.2f}%"
            )

    print("\nInterpretation:")
    print("- candidate_center_oracle_size isolates center/grid error; low IoU50 means center refinement is essential.")
    print("- oracle_center_median_size isolates box-size variability; high IoU50 means a simple sensor-conditioned size prior may suffice.")
    print("- candidate_center_median_size is the cheap non-learned geometry baseline.")


if __name__ == "__main__":
    main()
