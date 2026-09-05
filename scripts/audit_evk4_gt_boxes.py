"""PART 1 — audit EVK4 GT boxes vs enumerated 40-ms windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from orbitsight.evaluation.tii_style import load_tii_gt, tii_iou, windows_overlap
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows

SPLIT = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
)
EVK4_SEQ = "2025_12_23_21_12_28_EVK4_mag5.2"


def write_csv(path: Path, rows: list[dict]) -> None:
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
    parser.add_argument("--sequence", default=EVK4_SEQ)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/runs/2026-08-31/challenge_aligned_confidence"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seq = args.sequence
    arr = np.load(args.split_dir / f"{seq}_labeled_events.npy", mmap_mode="r")
    ts = np.asarray(arr[:, 3])
    t0_evt, t1_evt = int(ts[0]), int(ts[-1])
    window_starts = enumerate_challenge_windows(ts)

    gt_rows = load_tii_gt(args.split_dir / f"{seq}_bb_windows_40ms.txt")
    gt_in_windows = set()
    for wi, ws in enumerate(window_starts):
        we = int(ws) + WINDOW_US
        for gi, gt in enumerate(gt_rows):
            if windows_overlap(int(ws), we, gt[0], gt[1]):
                gt_in_windows.add(gi)

    unassigned = [gi for gi in range(len(gt_rows)) if gi not in gt_in_windows]
    audit_rows = []
    for gi in unassigned:
        ws, we, cx, cy, w, h = gt_rows[gi]
        overlaps_any = False
        overlap_ws = None
        for ws_enum in window_starts:
            we_enum = int(ws_enum) + WINDOW_US
            if windows_overlap(int(ws_enum), we_enum, ws, we):
                overlaps_any = True
                overlap_ws = int(ws_enum)
                break
        reason = []
        if ws < t0_evt or we > t1_evt + WINDOW_US:
            reason.append("outside_event_span")
        if not overlaps_any:
            reason.append("no_enumerated_window_overlap")
        if ws >= we:
            reason.append("invalid_window_bounds")
        audit_rows.append(
            {
                "gt_index": gi,
                "window_start_us": ws,
                "window_end_us": we,
                "center_x": cx,
                "center_y": cy,
                "width": w,
                "height": h,
                "event_first_us": t0_evt,
                "event_last_us": t1_evt,
                "overlaps_enumerated_window": overlaps_any,
                "example_enum_start_us": overlap_ws if overlap_ws is not None else "",
                "reason": "|".join(reason) if reason else "unknown",
            }
        )

    summary = {
        "sequence": seq,
        "total_gt_boxes": len(gt_rows),
        "gt_in_enumerated_windows": len(gt_in_windows),
        "unassigned_gt_boxes": len(unassigned),
        "enumerated_windows": len(window_starts),
        "event_first_us": t0_evt,
        "event_last_us": t1_evt,
        "enumeration_rule": "np.arange(t0, t1, 40000) half-open [ws, ws+40000)",
    }
    write_csv(args.out_dir / "evk4_gt_audit.csv", audit_rows)
    (args.out_dir / "evk4_gt_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"unassigned={len(unassigned)} rows={args.out_dir / 'evk4_gt_audit.csv'}")


if __name__ == "__main__":
    main()
