from __future__ import annotations

"""PART 3 — characterize all challenge 40-ms windows (Training_sets)."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows
from orbitsight.io import read_detection_file

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
    parser.add_argument("--out-dir", type=Path, default=Path("docs/runs/2026-08-30/challenge_metric_baseline"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Compare enumeration to TII visualizer convention (documented in report).
    convention = {
        "window_us": WINDOW_US,
        "start_rule": "np.arange(int(ev_t[0]), int(ev_t[-1]), WINDOW_US)",
        "interval": "half-open [ws, ws+WINDOW_US)",
        "source": "OrbitSight_DataLoader/visualize_dataset.py",
    }
    (args.out_dir / "window_convention.json").write_text(json.dumps(convention, indent=2), encoding="utf-8")

    seq_rows = []
    gt_count_hist = Counter()
    total_windows = 0
    windows_with_gt = 0
    empty_windows = 0
    n1 = n2 = n3p = 0
    by_sensor = defaultdict(lambda: {"windows": 0, "with_gt": 0, "empty": 0, "gt_boxes": 0})

    for gt_path in sorted(args.split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        sensor = sensor_name(sequence)
        arr = np.load(args.split_dir / f"{sequence}_labeled_events.npy", mmap_mode="r")
        ts = arr[:, 3]
        starts = enumerate_challenge_windows(ts)
        n_win = len(starts)

        gts = read_detection_file(gt_path)
        # Assign GT boxes to enumerated windows by temporal overlap (TII convention).
        starts_list = [int(s) for s in starts]
        ends_list = [s + WINDOW_US for s in starts_list]
        gt_counts = [0] * len(starts_list)
        for gt in gts:
            for i, (ws, we) in enumerate(zip(starts_list, ends_list)):
                if gt.start_us < we and gt.end_us > ws:
                    gt_counts[i] += 1

        seq_with_gt = 0
        seq_empty = 0
        seq_gt_boxes = len(gts)
        for k in gt_counts:
            gt_count_hist[k] += 1
            if k == 0:
                seq_empty += 1
                empty_windows += 1
            else:
                seq_with_gt += 1
                windows_with_gt += 1
                if k == 1:
                    n1 += 1
                elif k == 2:
                    n2 += 1
                else:
                    n3p += 1
            by_sensor[sensor]["windows"] += 1

        by_sensor[sensor]["with_gt"] += seq_with_gt
        by_sensor[sensor]["empty"] += seq_empty
        by_sensor[sensor]["gt_boxes"] += seq_gt_boxes
        total_windows += n_win
        seq_rows.append(
            {
                "sequence": sequence,
                "sensor": sensor,
                "n_windows": n_win,
                "windows_with_gt": seq_with_gt,
                "empty_windows": seq_empty,
                "gt_boxes": seq_gt_boxes,
                "t0": int(ts[0]) if len(ts) else -1,
                "t1": int(ts[-1]) if len(ts) else -1,
            }
        )
        print(f"{sequence}: windows={n_win} with_gt={seq_with_gt} empty={seq_empty}", flush=True)

    summary = {
        "total_windows": total_windows,
        "windows_with_gt": windows_with_gt,
        "empty_windows": empty_windows,
        "pct_with_gt": 100.0 * windows_with_gt / max(total_windows, 1),
        "pct_empty": 100.0 * empty_windows / max(total_windows, 1),
        "windows_1_gt": n1,
        "windows_2_gt": n2,
        "windows_3plus_gt": n3p,
        "pct_1_gt_among_positive": 100.0 * n1 / max(windows_with_gt, 1),
        "pct_2_gt_among_positive": 100.0 * n2 / max(windows_with_gt, 1),
        "pct_3plus_gt_among_positive": 100.0 * n3p / max(windows_with_gt, 1),
    }
    with (args.out_dir / "window_characterization_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    with (args.out_dir / "window_characterization_by_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
        w.writeheader()
        w.writerows(seq_rows)
    sensor_rows = [{"sensor": s, **v} for s, v in sorted(by_sensor.items())]
    with (args.out_dir / "window_characterization_by_sensor.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sensor_rows[0].keys()))
        w.writeheader()
        w.writerows(sensor_rows)
    hist_rows = [{"gt_count": k, "n_windows": v} for k, v in sorted(gt_count_hist.items())]
    with (args.out_dir / "window_gt_count_hist.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gt_count", "n_windows"])
        w.writeheader()
        w.writerows(hist_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
