from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from orbitsight.io import Detection, read_detection_file
from orbitsight.proposals import Candidate, RawGridProposer, infer_sensor_geometry

TOPKS = (1, 3, 5, 10, 20)


def candidate_rank(candidates: list[Candidate], gt: Detection, margin: float) -> int | None:
    """Return 1-based rank of first candidate spatially compatible with GT.

    This is a proposal-recall diagnostic, not the official IoU metric.  The one-cell
    margin is deliberately the same permissive criterion used during our early
    proposal forensics so the result answers: 'did the cheap front-end preserve a
    usable local region for a later learned scorer/refiner?'
    """
    for rank, candidate in enumerate(candidates, start=1):
        if (
            abs(candidate.cx - gt.cx) <= gt.width / 2.0 + margin
            and abs(candidate.cy - gt.cy) <= gt.height / 2.0 + margin
        ):
            return rank
    return None


def evaluate_sequence(npy_path: Path, gt_path: Path, sequence: str, top_k: int) -> dict[str, float | int | str]:
    arr = np.load(npy_path, mmap_mode="r")
    timestamps = arr[:, 3]
    width, height, cell = infer_sensor_geometry(sequence)
    proposer = RawGridProposer(width, height, cell, top_k=top_k)

    grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
    for gt in read_detection_file(gt_path):
        grouped[(gt.start_us, gt.end_us)].append(gt)

    hits = {k: 0 for k in TOPKS}
    total_targets = 0
    missing_from_pool = 0
    window_ms: list[float] = []

    for (start_us, end_us), gts in sorted(grouped.items()):
        t0 = perf_counter_ns()
        left = int(np.searchsorted(timestamps, start_us, side="left"))
        right = int(np.searchsorted(timestamps, end_us, side="left"))
        events = np.asarray(arr[left:right, :4])
        candidates = proposer.propose(events)
        t1 = perf_counter_ns()
        window_ms.append((t1 - t0) / 1_000_000.0)

        for gt in gts:
            total_targets += 1
            rank = candidate_rank(candidates, gt, margin=float(cell))
            if rank is None:
                missing_from_pool += 1
                continue
            for k in TOPKS:
                if rank <= k:
                    hits[k] += 1

    row: dict[str, float | int | str] = {
        "sequence": sequence,
        "sensor": "DAVIS" if sequence.upper().startswith("DAVIS") else ("DVX" if sequence.upper().startswith("DVX") else "EVK4"),
        "targets": total_targets,
        "gt_windows": len(grouped),
        "missing_from_pool": missing_from_pool,
        "mean_window_ms": float(np.mean(window_ms)) if window_ms else float("nan"),
        "p95_window_ms": float(np.percentile(window_ms, 95)) if window_ms else float("nan"),
    }
    for k in TOPKS:
        row[f"top{k}_recall"] = hits[k] / total_targets if total_targets else float("nan")
        row[f"top{k}_hits"] = hits[k]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure B1 raw sparse proposal recall on labelled training sequences")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", default="b1_raw_proposal_results.csv")
    args = parser.parse_args()

    if args.top_k < max(TOPKS):
        raise SystemExit(f"--top-k must be at least {max(TOPKS)} for the standard report")

    split_dir = Path(args.split_dir)
    rows: list[dict[str, float | int | str]] = []

    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        npy_path = split_dir / f"{sequence}_labeled_events.npy"
        if not npy_path.exists():
            raise FileNotFoundError(npy_path)
        row = evaluate_sequence(npy_path, gt_path, sequence, args.top_k)
        rows.append(row)
        print(
            f"{sequence:68s} targets={int(row['targets']):5d} "
            f"T1={100*float(row['top1_recall']):6.2f}% "
            f"T3={100*float(row['top3_recall']):6.2f}% "
            f"T5={100*float(row['top5_recall']):6.2f}% "
            f"T20={100*float(row['top20_recall']):6.2f}% "
            f"p95={float(row['p95_window_ms']):7.3f} ms"
        )

    if not rows:
        raise SystemExit("No *_bb_windows_40ms.txt files found")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_targets = sum(int(r["targets"]) for r in rows)
    print("\n=== B1 RAW PROPOSAL SUMMARY ===")
    print(f"sequences={len(rows)} targets={total_targets}")
    for k in TOPKS:
        hits = sum(int(r[f"top{k}_hits"]) for r in rows)
        micro = hits / total_targets if total_targets else float("nan")
        macro = float(np.mean([float(r[f"top{k}_recall"]) for r in rows]))
        print(f"top{k}_micro_recall={100*micro:.3f}%")
        print(f"top{k}_macro_recall={100*macro:.3f}%")
    weighted_mean_ms = sum(float(r["mean_window_ms"]) * int(r["gt_windows"]) for r in rows) / max(sum(int(r["gt_windows"]) for r in rows), 1)
    print(f"mean_gt_window_pipeline_ms={weighted_mean_ms:.4f}")
    print(f"csv={out_path}")
    print("NOTE: proposal recall is NOT AP/F1/IoU. It measures whether later local AI receives the target region.")


if __name__ == "__main__":
    main()
