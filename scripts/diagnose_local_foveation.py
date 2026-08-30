from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from orbitsight.io import Detection, read_detection_file
from orbitsight.proposals import RawGridProposer, infer_sensor_geometry

METHODS = ("C0_grid", "C1_centroid", "C2_recent_centroid", "C3_linear_motion")


def iou_centre(cx: float, cy: float, w: float, h: float, gt: Detection) -> float:
    ax1, ay1 = cx - w / 2.0, cy - h / 2.0
    ax2, ay2 = cx + w / 2.0, cy + h / 2.0
    bx1, by1 = gt.cx - gt.width / 2.0, gt.cy - gt.height / 2.0
    bx2, by2 = gt.cx + gt.width / 2.0, gt.cy + gt.height / 2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w * h + gt.width * gt.height - inter
    return inter / union if union > 0 else 0.0


def gt_compatible(candidates, gt: Detection, margin: float):
    for candidate in candidates:
        if (
            abs(candidate.cx - gt.cx) <= gt.width / 2.0 + margin
            and abs(candidate.cy - gt.cy) <= gt.height / 2.0 + margin
        ):
            return candidate
    return None


def roi_mask(events: np.ndarray, cx: float, cy: float, cell: float) -> np.ndarray:
    half = 2.0 * cell
    x = events[:, 0]
    y = events[:, 1]
    return (x >= cx - half) & (x <= cx + half) & (y >= cy - half) & (y <= cy + half)


def estimate_all(events: np.ndarray, candidate, cell: float, start_us: int, end_us: int) -> dict[str, tuple[float, float]]:
    cx0, cy0 = candidate.cx, candidate.cy
    out: dict[str, tuple[float, float]] = {"C0_grid": (cx0, cy0)}
    mask = roi_mask(events, cx0, cy0, cell)
    roi = events[mask]
    if len(roi) == 0:
        out["C1_centroid"] = (cx0, cy0)
        out["C2_recent_centroid"] = (cx0, cy0)
        out["C3_linear_motion"] = (cx0, cy0)
        return out

    out["C1_centroid"] = (float(np.mean(roi[:, 0])), float(np.mean(roi[:, 1])))

    if roi.shape[1] >= 4 and end_us > start_us:
        t = roi[:, 3].astype(np.float64)
        t0, t1 = float(t.min()), float(t.max())
        norm = (t - t0) / max(t1 - t0, 1.0)
        weights = 0.25 + 0.75 * norm
        wsum = float(weights.sum())
        out["C2_recent_centroid"] = (
            float(np.sum(roi[:, 0] * weights) / wsum),
            float(np.sum(roi[:, 1] * weights) / wsum),
        )
        if len(roi) >= 4:
            duration = max(end_us - start_us, 1.0)
            mid = (start_us + end_us) / 2.0
            tn = (t - mid) / duration
            ax, bx = np.polyfit(tn, roi[:, 0], 1)
            ay, by = np.polyfit(tn, roi[:, 1], 1)
            out["C3_linear_motion"] = (float(bx), float(by))
        else:
            out["C3_linear_motion"] = out["C1_centroid"]
    else:
        out["C2_recent_centroid"] = out["C1_centroid"]
        out["C3_linear_motion"] = out["C1_centroid"]
    return out


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed centre-estimator localization diagnostic (training only)")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--folds", default="sequence_folds.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-dir", default="docs/runs/2026-08-30")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    seq_to_fold: dict[str, int] = {}
    for fold in folds:
        for seq in fold["validation"]:
            seq_to_fold[seq] = int(fold["fold"])

    records: list[dict] = []
    timings: dict[str, list[float]] = defaultdict(list)

    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
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

        for (start_us, end_us), gts in sorted(grouped.items()):
            left = int(np.searchsorted(timestamps, start_us, side="left"))
            right = int(np.searchsorted(timestamps, end_us, side="left"))
            events = np.asarray(arr[left:right, :4])
            candidates = proposer.propose(events)
            for gt in gts:
                candidate = gt_compatible(candidates, gt, margin=float(cell))
                if candidate is None:
                    continue
                t0 = perf_counter_ns()
                centres = estimate_all(events, candidate, cell, start_us, end_us)
                elapsed_ms = (perf_counter_ns() - t0) / 1_000_000.0
                for method in METHODS:
                    cx, cy = centres[method]
                    err = math.hypot(cx - gt.cx, cy - gt.cy)
                    iou = iou_centre(cx, cy, gt.width, gt.height, gt)
                    records.append(
                        {
                            "sequence": sequence,
                            "sensor": sensor,
                            "fold": seq_to_fold.get(sequence, -1),
                            "method": method,
                            "centre_error": err,
                            "iou_oracle_size": iou,
                            "iou50": float(iou >= 0.5),
                            "compute_ms": elapsed_ms,
                        }
                    )
                    timings[method].append(elapsed_ms)

    def aggregate(rows: list[dict], key: str, value: str) -> list[dict]:
        grouped_vals: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped_vals[(row[key], row["method"])].append(row)
        out = []
        for (group_val, method), items in sorted(grouped_vals.items()):
            errs = [float(r["centre_error"]) for r in items]
            ious = [float(r["iou_oracle_size"]) for r in items]
            comp = [float(r["compute_ms"]) for r in items]
            stats_e = summarize(errs)
            stats_c = summarize(comp)
            out.append(
                {
                    key: group_val,
                    "method": method,
                    "n": len(items),
                    "centre_error_mean": stats_e["mean"],
                    "centre_error_median": stats_e["median"],
                    "centre_error_p90": stats_e["p90"],
                    "iou_oracle_mean": float(np.mean(ious)),
                    "iou50_pct": 100.0 * float(np.mean([float(r["iou50"]) for r in items])),
                    "compute_p50_ms": float(np.percentile(comp, 50)),
                    "compute_p95_ms": float(np.percentile(comp, 95)),
                }
            )
        return out

    sensor_rows = aggregate(records, "sensor", "sensor")
    sequence_rows = aggregate(records, "sequence", "sequence")
    fold_rows = aggregate([r for r in records if int(r["fold"]) >= 0], "fold", "fold")

    for name, rows in (
        ("local_foveation_by_sensor.csv", sensor_rows),
        ("local_foveation_by_sequence.csv", sequence_rows),
        ("local_foveation_by_fold.csv", fold_rows),
    ):
        if rows:
            with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    md = out_dir.parent / "2026-08-30_local_foveation_diagnostic.md"
    lines = [
        "# Local foveation diagnostic — 2026-08-30",
        "",
        "Training data only. GT used for evaluation only, not inside estimators.",
        "",
        "## By sensor",
        "",
        "| sensor | method | n | centre err mean | median | p90 | IoU oracle mean | IoU>=0.5 % | compute p50 ms | p95 ms |",
        "|--------|--------|---:|----------------:|-------:|----:|----------------:|------------:|---------------:|-------:|",
    ]
    for row in sensor_rows:
        lines.append(
            f"| {row['sensor']} | {row['method']} | {row['n']} | {row['centre_error_mean']:.3f} | "
            f"{row['centre_error_median']:.3f} | {row['centre_error_p90']:.3f} | {row['iou_oracle_mean']:.4f} | "
            f"{row['iou50_pct']:.2f} | {row['compute_p50_ms']:.4f} | {row['compute_p95_ms']:.4f} |"
        )
    lines.extend(["", "## By sequence (macro over methods)", ""])
    for row in sequence_rows:
        lines.append(
            f"- `{row['sequence']}` {row['method']}: err_mean={row['centre_error_mean']:.3f}, "
            f"iou50={row['iou50_pct']:.2f}%, compute_p95={row['compute_p95_ms']:.4f} ms"
        )
    lines.extend(["", "## Fold variation (validation sequences only)", ""])
    for row in fold_rows:
        lines.append(
            f"- fold {row['fold']} {row['method']}: err_mean={row['centre_error_mean']:.3f}, "
            f"iou50={row['iou50_pct']:.2f}%"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"records={len(records)} md={md}")


if __name__ == "__main__":
    main()
