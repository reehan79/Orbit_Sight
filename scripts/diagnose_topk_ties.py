from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from orbitsight.io import Detection, read_detection_file
from orbitsight.proposals import Candidate, RawGridProposer, infer_sensor_geometry

TOPKS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class GridState:
    counts: np.ndarray
    grid_w: int
    cell_size: int
    width: int
    height: int
    occupied: np.ndarray


def build_grid(events: np.ndarray, width: int, height: int, cell_size: int) -> GridState | None:
    if len(events) == 0:
        return None
    x = events[:, 0].astype(np.int64, copy=False)
    y = events[:, 1].astype(np.int64, copy=False)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return None
    grid_w = int(np.ceil(width / cell_size))
    gx, gy = x // cell_size, y // cell_size
    ids = gy * grid_w + gx
    counts = np.bincount(ids, minlength=grid_w * int(np.ceil(height / cell_size)))
    occupied = np.flatnonzero(counts)
    if len(occupied) == 0:
        return None
    return GridState(counts=counts, grid_w=grid_w, cell_size=cell_size, width=width, height=height, occupied=occupied)


def cells_to_candidates(grid: GridState, ranked_ids: np.ndarray) -> list[Candidate]:
    if len(ranked_ids) == 0:
        return []
    max_count = max(int(grid.counts[ranked_ids[0]]), 1)
    out: list[Candidate] = []
    for gid in ranked_ids:
        gy_i, gx_i = divmod(int(gid), grid.grid_w)
        count = int(grid.counts[gid])
        out.append(
            Candidate(
                gx_i * grid.cell_size + grid.cell_size / 2.0,
                gy_i * grid.cell_size + grid.cell_size / 2.0,
                count / max_count,
                count,
                gx_i,
                gy_i,
            )
        )
    return out


def ranked_ids_argpartition(grid: GridState, top_k: int) -> np.ndarray:
    k = min(top_k, len(grid.occupied))
    if len(grid.occupied) > k:
        local = np.argpartition(grid.counts[grid.occupied], -k)[-k:]
        ranked = grid.occupied[local]
    else:
        ranked = grid.occupied
    ranked = ranked[np.argsort(grid.counts[ranked])[::-1]]
    return ranked


def ranked_ids_deterministic(grid: GridState, top_k: int) -> np.ndarray:
    k = min(top_k, len(grid.occupied))
    order = np.lexsort((grid.occupied, -grid.counts[grid.occupied]))
    return grid.occupied[order[:k]]


def boundary_stats(grid: GridState, top_k: int) -> dict[str, int | float | bool]:
    det = ranked_ids_deterministic(grid, top_k)
    k = len(det)
    boundary_count = int(grid.counts[det[-1]]) if k else 0
    n_strictly_above = int(np.sum(grid.counts[grid.occupied] > boundary_count))
    n_tied = int(np.sum(grid.counts[grid.occupied] == boundary_count))
    remaining_slots = max(top_k - n_strictly_above, 0)
    tied_fit = min(n_tied, remaining_slots) if k >= top_k else n_tied
    cur = ranked_ids_argpartition(grid, top_k)
    differs = not np.array_equal(cur, det)
    return {
        "occupied_cells": int(len(grid.occupied)),
        "boundary_count": boundary_count,
        "tied_at_boundary": n_tied,
        "tied_fit_in_topk": int(tied_fit),
        "remaining_topk_slots_for_boundary_ties": int(remaining_slots),
        "current_differs_from_deterministic": bool(differs),
    }


def gt_compatible_cell_id(grid: GridState, gt: Detection, margin: float) -> int | None:
    half_w = gt.width / 2.0 + margin
    half_h = gt.height / 2.0 + margin
    for gid in grid.occupied:
        gy_i, gx_i = divmod(int(gid), grid.grid_w)
        cx = gx_i * grid.cell_size + grid.cell_size / 2.0
        cy = gy_i * grid.cell_size + grid.cell_size / 2.0
        if abs(cx - gt.cx) <= half_w and abs(cy - gt.cy) <= half_h:
            return int(gid)
    return None


def gt_rank(candidates: list[Candidate], gt: Detection, margin: float) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if (
            abs(candidate.cx - gt.cx) <= gt.width / 2.0 + margin
            and abs(candidate.cy - gt.cy) <= gt.height / 2.0 + margin
        ):
            return rank
    return None


def gt_affected_by_boundary_tie(grid: GridState, gt: Detection, margin: float, top_k: int) -> bool:
    gid = gt_compatible_cell_id(grid, gt, margin)
    if gid is None:
        return False
    stats = boundary_stats(grid, top_k)
    if stats["tied_at_boundary"] <= 1:
        return False
    boundary_count = int(stats["boundary_count"])
    if int(grid.counts[gid]) != boundary_count:
        return False
    det = ranked_ids_deterministic(grid, top_k)
    cur = ranked_ids_argpartition(grid, top_k)
    in_det = gid in set(det.tolist())
    in_cur = gid in set(cur.tolist())
    return in_det != in_cur


def propose_mode(grid: GridState, top_k: int, mode: str) -> list[Candidate]:
    if mode == "argpartition":
        ranked = ranked_ids_argpartition(grid, top_k)
    elif mode == "deterministic":
        ranked = ranked_ids_deterministic(grid, top_k)
    else:
        raise ValueError(mode)
    return cells_to_candidates(grid, ranked)


def evaluate_mode(split_dir: Path, top_k: int, mode: str) -> tuple[list[dict], list[float]]:
    rows: list[dict] = []
    latencies: list[float] = []
    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        npy_path = split_dir / f"{sequence}_labeled_events.npy"
        if not npy_path.exists():
            raise FileNotFoundError(npy_path)
        arr = np.load(npy_path, mmap_mode="r")
        timestamps = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
        for gt in read_detection_file(gt_path):
            grouped[(gt.start_us, gt.end_us)].append(gt)
        hits = {k: 0 for k in TOPKS}
        total_targets = 0
        for (_start, _end), gts in sorted(grouped.items()):
            if not gts:
                continue
            start_us, end_us = _start, _end
            t0 = perf_counter_ns()
            left = int(np.searchsorted(timestamps, start_us, side="left"))
            right = int(np.searchsorted(timestamps, end_us, side="left"))
            events = np.asarray(arr[left:right, :4])
            grid = build_grid(events, width, height, cell)
            if grid is None:
                candidates = []
            else:
                candidates = propose_mode(grid, top_k, mode)
            latencies.append((perf_counter_ns() - t0) / 1_000_000.0)
            for gt in gts:
                total_targets += 1
                rank = gt_rank(candidates, gt, margin=float(cell))
                if rank is None:
                    continue
                for k in TOPKS:
                    if rank <= k:
                        hits[k] += 1
        row = {
            "sequence": sequence,
            "sensor": "DAVIS" if sequence.upper().startswith("DAVIS") else ("DVX" if sequence.upper().startswith("DVX") else "EVK4"),
            "targets": total_targets,
        }
        for k in TOPKS:
            row[f"top{k}_recall"] = hits[k] / total_targets if total_targets else float("nan")
            row[f"top{k}_hits"] = hits[k]
        rows.append(row)
    return rows, latencies


def aggregate_recall(rows: list[dict]) -> dict[str, float]:
    total_targets = sum(int(r["targets"]) for r in rows)
    out: dict[str, float] = {"targets": float(total_targets)}
    for k in TOPKS:
        hits = sum(int(r[f"top{k}_hits"]) for r in rows)
        out[f"top{k}_micro"] = hits / total_targets if total_targets else float("nan")
        out[f"top{k}_macro"] = float(np.mean([float(r[f"top{k}_recall"]) for r in rows]))
    return out


def diagnose_windows(split_dir: Path, top_k: int) -> tuple[list[dict], dict[str, float]]:
    window_rows: list[dict] = []
    seq_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        npy_path = split_dir / f"{sequence}_labeled_events.npy"
        arr = np.load(npy_path, mmap_mode="r")
        timestamps = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
        for gt in read_detection_file(gt_path):
            grouped[(gt.start_us, gt.end_us)].append(gt)

        for (start_us, end_us), gts in sorted(grouped.items()):
            left = int(np.searchsorted(timestamps, start_us, side="left"))
            right = int(np.searchsorted(timestamps, end_us, side="left"))
            events = np.asarray(arr[left:right, :4])
            grid = build_grid(events, width, height, cell)
            if grid is None:
                continue
            stats = boundary_stats(grid, top_k)
            gt_tie = False
            for gt in gts:
                if gt_affected_by_boundary_tie(grid, gt, margin=float(cell), top_k=top_k):
                    gt_tie = True
                    break
            row = {
                "sequence": sequence,
                "sensor": "DAVIS" if sequence.upper().startswith("DAVIS") else ("DVX" if sequence.upper().startswith("DVX") else "EVK4"),
                "window_start_us": start_us,
                "window_end_us": end_us,
                "gt_targets": len(gts),
                **stats,
                "gt_compatible_affected_by_boundary_tie": gt_tie,
            }
            window_rows.append(row)
            seq_totals[sequence]["windows"] += 1
            if stats["current_differs_from_deterministic"]:
                seq_totals[sequence]["differs"] += 1
            if stats["tied_at_boundary"] > 1:
                seq_totals[sequence]["boundary_tie_windows"] += 1
            if gt_tie:
                seq_totals[sequence]["gt_tie_affected"] += 1

    aggregate = {
        "windows": len(window_rows),
        "windows_current_differs": sum(int(r["current_differs_from_deterministic"]) for r in window_rows),
        "windows_boundary_tie": sum(int(r["tied_at_boundary"] > 1) for r in window_rows),
        "windows_gt_tie_affected": sum(int(r["gt_compatible_affected_by_boundary_tie"]) for r in window_rows),
    }
    return window_rows, aggregate


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Top-K tie behavior for RawGridProposer")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-dir", default="docs/runs/2026-08-30")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_rows, window_agg = diagnose_windows(split_dir, args.top_k)
    write_csv(out_dir / "topk_tie_windows.csv", window_rows)

    seq_rows = []
    seq_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in window_rows:
        seq = row["sequence"]
        seq_counts[seq]["windows"] += 1
        seq_counts[seq]["occupied_cells"] += int(row["occupied_cells"])
        if row["current_differs_from_deterministic"]:
            seq_counts[seq]["differs"] += 1
        if int(row["tied_at_boundary"]) > 1:
            seq_counts[seq]["boundary_tie_windows"] += 1
        if row["gt_compatible_affected_by_boundary_tie"]:
            seq_counts[seq]["gt_tie_affected"] += 1
    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        counts = seq_counts[sequence]
        n = max(counts["windows"], 1)
        seq_rows.append(
            {
                "sequence": sequence,
                "sensor": "DAVIS" if sequence.upper().startswith("DAVIS") else ("DVX" if sequence.upper().startswith("DVX") else "EVK4"),
                "gt_windows": counts["windows"],
                "mean_occupied_cells": counts["occupied_cells"] / n,
                "windows_current_differs": counts["differs"],
                "windows_boundary_tie": counts["boundary_tie_windows"],
                "windows_gt_tie_affected": counts["gt_tie_affected"],
            }
        )
    write_csv(out_dir / "topk_tie_per_sequence.csv", seq_rows)

    rows_a, lat_a = evaluate_mode(split_dir, args.top_k, "argpartition")
    rows_b, lat_b = evaluate_mode(split_dir, args.top_k, "deterministic")
    agg_a = aggregate_recall(rows_a)
    agg_b = aggregate_recall(rows_b)

    compare_rows = []
    for key in ("top1_micro", "top3_micro", "top5_micro", "top10_micro", "top20_micro", "top1_macro", "top3_macro", "top5_macro", "top10_macro", "top20_macro"):
        compare_rows.append({"metric": key, "argpartition": agg_a[key], "deterministic": agg_b[key], "delta_pp": 100.0 * (agg_b[key] - agg_a[key])})
    lat_row = {
        "metric": "proposal_latency_ms",
        "argpartition_p50": float(np.percentile(lat_a, 50)) if lat_a else float("nan"),
        "argpartition_p95": float(np.percentile(lat_a, 95)) if lat_a else float("nan"),
        "argpartition_p99": float(np.percentile(lat_a, 99)) if lat_a else float("nan"),
        "deterministic_p50": float(np.percentile(lat_b, 50)) if lat_b else float("nan"),
        "deterministic_p95": float(np.percentile(lat_b, 95)) if lat_b else float("nan"),
        "deterministic_p99": float(np.percentile(lat_b, 99)) if lat_b else float("nan"),
    }
    write_csv(out_dir / "topk_tie_recall_compare.csv", compare_rows)

    # per-sequence compare
    per_seq = []
    map_a = {r["sequence"]: r for r in rows_a}
    map_b = {r["sequence"]: r for r in rows_b}
    for sequence in sorted(map_a):
        ra, rb = map_a[sequence], map_b[sequence]
        item = {"sequence": sequence, "sensor": ra["sensor"], "targets": ra["targets"]}
        for k in TOPKS:
            item[f"top{k}_argpartition"] = ra[f"top{k}_recall"]
            item[f"top{k}_deterministic"] = rb[f"top{k}_recall"]
            item[f"top{k}_delta_pp"] = 100.0 * (float(rb[f"top{k}_recall"]) - float(ra[f"top{k}_recall"]))
        per_seq.append(item)
    write_csv(out_dir / "topk_tie_recall_compare_per_sequence.csv", per_seq)

    with (out_dir / "topk_tie_aggregate.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"gt_windows={window_agg['windows']}\n")
        handle.write(f"windows_current_differs={window_agg['windows_current_differs']}\n")
        handle.write(f"windows_boundary_tie={window_agg['windows_boundary_tie']}\n")
        handle.write(f"windows_gt_tie_affected={window_agg['windows_gt_tie_affected']}\n")
        handle.write(f"top20_micro_argpartition={100*agg_a['top20_micro']:.6f}%\n")
        handle.write(f"top20_micro_deterministic={100*agg_b['top20_micro']:.6f}%\n")
        handle.write(f"top20_micro_delta_pp={100*(agg_b['top20_micro']-agg_a['top20_micro']):.6f}\n")
        for label, lat in (("argpartition", lat_a), ("deterministic", lat_b)):
            handle.write(f"{label}_p50_ms={np.percentile(lat,50):.6f}\n")
            handle.write(f"{label}_p95_ms={np.percentile(lat,95):.6f}\n")
            handle.write(f"{label}_p99_ms={np.percentile(lat,99):.6f}\n")

    print("=== TOP-K TIE DIAGNOSTIC ===")
    print(f"gt_windows={window_agg['windows']}")
    print(f"windows_current_differs={window_agg['windows_current_differs']}")
    print(f"windows_boundary_tie={window_agg['windows_boundary_tie']}")
    print(f"windows_gt_tie_affected={window_agg['windows_gt_tie_affected']}")
    print(f"top20_micro argpartition={100*agg_a['top20_micro']:.4f}% deterministic={100*agg_b['top20_micro']:.4f}% delta={100*(agg_b['top20_micro']-agg_a['top20_micro']):+.4f}pp")
    print(f"latency_p50 argpartition={np.percentile(lat_a,50):.4f}ms deterministic={np.percentile(lat_b,50):.4f}ms")
    print(f"latency_p95 argpartition={np.percentile(lat_a,95):.4f}ms deterministic={np.percentile(lat_b,95):.4f}ms")
    print(f"latency_p99 argpartition={np.percentile(lat_a,99):.4f}ms deterministic={np.percentile(lat_b,99):.4f}ms")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
