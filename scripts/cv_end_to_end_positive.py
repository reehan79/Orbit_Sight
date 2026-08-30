from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from orbitsight.evaluation.detection_aggregate import aggregate_detection_metrics, failure_buckets
from orbitsight.features import (
    FEATURE_NAMES,
    LOCAL_GEOMETRY_NAMES,
    extract_candidate_features,
    extract_local_geometry_features,
    refine_c1_centroid,
)
from orbitsight.io import Detection, read_detection_file
from orbitsight.models import fit_rankers, score_ranker
from orbitsight.proposals import Candidate, RawGridProposer, infer_sensor_geometry

PRIOR_MS = 80
TOP_K = 20


@dataclass(frozen=True)
class EvalConfig:
    label: str
    ranker: str | None
    centre: str
    size: str


def load_table(path: Path) -> dict[str, np.ndarray]:
    sequences, sensors, starts, ends, ranks, targets = [], [], [], [], [], []
    features, bbox_targets = [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequences.append(row["sequence"])
            sensors.append(row["sensor"])
            starts.append(int(row["window_start_us"]))
            ends.append(int(row["window_end_us"]))
            ranks.append(int(row["candidate_rank"]))
            targets.append(int(row["target"]))
            features.append([float(row[n]) for n in FEATURE_NAMES])
            if row["bbox_log_w_cells"] == "":
                bbox_targets.append([np.nan, np.nan])
            else:
                bbox_targets.append([float(row["bbox_log_w_cells"]), float(row["bbox_log_h_cells"])])
    return {
        "sequence": np.asarray(sequences, dtype=object),
        "sensor": np.asarray(sensors, dtype=object),
        "start": np.asarray(starts, dtype=np.int64),
        "end": np.asarray(ends, dtype=np.int64),
        "rank": np.asarray(ranks, dtype=np.int16),
        "target": np.asarray(targets, dtype=np.int8),
        "X": np.asarray(features, dtype=np.float32),
        "bbox_log_wh": np.asarray(bbox_targets, dtype=np.float32),
    }


def compatible(candidate: Candidate, gt: Detection, margin: float) -> bool:
    return (
        abs(candidate.cx - gt.cx) <= gt.width / 2.0 + margin
        and abs(candidate.cy - gt.cy) <= gt.height / 2.0 + margin
    )


def iou_box(cx: float, cy: float, w: float, h: float, gt: Detection) -> float:
    ax1, ay1 = cx - w / 2.0, cy - h / 2.0
    ax2, ay2 = cx + w / 2.0, cy + h / 2.0
    bx1, by1 = gt.cx - gt.width / 2.0, gt.cy - gt.height / 2.0
    bx2, by2 = gt.cx + gt.width / 2.0, gt.cy + gt.height / 2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w * h + gt.width * gt.height - inter
    return inter / union if union > 0 else 0.0


def sensor_name(sequence: str) -> str:
    u = sequence.upper()
    if u.startswith("DAVIS"):
        return "DAVIS"
    if u.startswith("DVX"):
        return "DVX"
    return "EVK4"


def compute_sensor_medians(split_dir: Path, train_sequences: set[str]) -> dict[str, tuple[float, float]]:
    widths: dict[str, list[float]] = defaultdict(list)
    heights: dict[str, list[float]] = defaultdict(list)
    all_w: list[float] = []
    all_h: list[float] = []
    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        if sequence not in train_sequences:
            continue
        sensor = sensor_name(sequence)
        for gt in read_detection_file(gt_path):
            widths[sensor].append(float(gt.width))
            heights[sensor].append(float(gt.height))
            all_w.append(float(gt.width))
            all_h.append(float(gt.height))
    medians = {s: (float(np.median(widths[s])), float(np.median(heights[s]))) for s in widths}
    if all_w:
        medians["_GLOBAL"] = (float(np.median(all_w)), float(np.median(all_h)))
    return medians


def fit_size_ridge(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([("scale", StandardScaler()), ("regressor", Ridge(alpha=2.0))])
    model.fit(X, y)
    return model


def fit_size_extratrees(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=32,
        max_depth=12,
        min_samples_leaf=24,
        max_features=None,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def build_size_training(
    table: dict[str, np.ndarray],
    train_idx: np.ndarray,
    split_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y_rows: list[list[float]] = []
    pos_idx = train_idx[table["target"][train_idx] == 1]
    by_window: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for idx in pos_idx:
        by_window[(str(table["sequence"][idx]), int(table["start"][idx]), int(table["end"][idx]))].append(int(idx))

    for (sequence, start, end), indices in by_window.items():
        npy_path = split_dir / f"{sequence}_labeled_events.npy"
        arr = np.load(npy_path, mmap_mode="r")
        timestamps = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="left"))
        prior_left = int(np.searchsorted(timestamps, start - PRIOR_MS * 1000, side="left"))
        current = np.asarray(arr[left:right, :4])
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        candidates = proposer.propose(current)
        if not candidates:
            continue
        for idx in indices:
            rank = int(table["rank"][idx])
            if rank < 1 or rank > len(candidates):
                continue
            candidate = candidates[rank - 1]
            feat15 = table["X"][idx]
            rcx, rcy = refine_c1_centroid(current, candidate.cx, candidate.cy, cell)
            local18 = extract_local_geometry_features(current, rcx, rcy, cell, width, height)
            X_rows.append(np.concatenate([feat15, local18]))
            y_rows.append(table["bbox_log_wh"][idx].tolist())

    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def predict_size(
    size_mode: str,
    sensor: str,
    sensor_medians: dict[str, tuple[float, float]],
    size_ridge,
    size_trees,
    size_X: np.ndarray,
    cell: float,
) -> tuple[float, float]:
    if size_mode == "S0":
        w, h = sensor_medians.get(sensor, sensor_medians["_GLOBAL"])
        return w, h
    if size_mode == "S1":
        log_wh = size_ridge.predict(size_X.reshape(1, -1))[0]
        return math.exp(float(log_wh[0])) * cell, math.exp(float(log_wh[1])) * cell
    if size_mode == "S2":
        log_wh = size_trees.predict(size_X.reshape(1, -1))[0]
        return math.exp(float(log_wh[0])) * cell, math.exp(float(log_wh[1])) * cell
    raise ValueError(size_mode)


def select_candidate(
    config: EvalConfig,
    candidates: list[Candidate],
    features: np.ndarray,
    ranker_bundle,
) -> Candidate:
    if not candidates:
        raise RuntimeError("empty candidates")
    if config.ranker is None:
        return candidates[0]
    ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
    scores = score_ranker(ranker_bundle, features, ranks)
    return candidates[int(np.argmax(scores))]


def pick_centre(config: EvalConfig, events: np.ndarray, candidate: Candidate, cell: float) -> tuple[float, float]:
    if config.centre == "C0":
        return candidate.cx, candidate.cy
    if config.centre == "C1":
        return refine_c1_centroid(events, candidate.cx, candidate.cy, cell)
    raise ValueError(config.centre)


def evaluate_fold(
    fold_id: int,
    train_sequences: set[str],
    val_sequences: set[str],
    table: dict[str, np.ndarray],
    split_dir: Path,
    configs: list[EvalConfig],
) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    train_idx = np.flatnonzero(np.isin(table["sequence"], list(train_sequences)))
    ranker_names = sorted({c.ranker for c in configs if c.ranker is not None})
    rankers = fit_rankers(
        table["X"][train_idx],
        table["target"][train_idx],
        table["rank"][train_idx],
        model_names=ranker_names,
    )
    sensor_medians = compute_sensor_medians(split_dir, train_sequences)
    size_X_train, size_y_train = build_size_training(table, train_idx, split_dir)
    size_ridge = fit_size_ridge(size_X_train, size_y_train)
    size_trees = fit_size_extratrees(size_X_train, size_y_train)

    fold_rows: list[dict] = []
    seq_rows: dict[str, list[dict]] = defaultdict(list)
    detail_by_config: dict[str, list[dict]] = {c.label: [] for c in configs}

    for config in configs:
        bundle = rankers[config.ranker] if config.ranker else None
        gt_records: list[dict] = []
        window_times: list[float] = []

        for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
            sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
            if sequence not in val_sequences:
                continue
            npy_path = split_dir / f"{sequence}_labeled_events.npy"
            arr = np.load(npy_path, mmap_mode="r")
            timestamps = arr[:, 3]
            width, height, cell = infer_sensor_geometry(sequence)
            sensor = sensor_name(sequence)
            proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
            grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
            for gt in read_detection_file(gt_path):
                grouped[(gt.start_us, gt.end_us)].append(gt)

            for (start_us, end_us), gts in sorted(grouped.items()):
                t0 = perf_counter_ns()
                left = int(np.searchsorted(timestamps, start_us, side="left"))
                right = int(np.searchsorted(timestamps, end_us, side="left"))
                prior_left = int(np.searchsorted(timestamps, start_us - PRIOR_MS * 1000, side="left"))
                current = np.asarray(arr[left:right, :4])
                prior = np.asarray(arr[prior_left:left, :4])
                candidates = proposer.propose(current)
                features = extract_candidate_features(current, prior, candidates, width, height, cell)
                if not candidates:
                    selected = None
                    cx = cy = 0.0
                    pred_w = pred_h = 0.0
                    feat_idx = None
                else:
                    selected = select_candidate(config, candidates, features, bundle)
                    cx, cy = pick_centre(config, current, selected, cell)
                    sel_rank = candidates.index(selected) + 1
                    feat_idx = sel_rank - 1
                    feat15 = features[feat_idx]
                    size_X = feat15
                    if config.size != "S0":
                        local18 = extract_local_geometry_features(current, cx, cy, cell, width, height)
                        size_X = np.concatenate([feat15, local18])
                    pred_w, pred_h = predict_size(
                        config.size, sensor, sensor_medians, size_ridge, size_trees, size_X, cell
                    )
                infer_ms = (perf_counter_ns() - t0) / 1_000_000.0
                window_times.append(infer_ms)

                for gt in gts:
                    proposal_hit = any(compatible(c, gt, float(cell)) for c in candidates)
                    ranker_hit = selected is not None and compatible(selected, gt, float(cell))
                    centre_err = math.hypot(cx - gt.cx, cy - gt.cy) if candidates else float("nan")
                    iou = iou_box(cx, cy, pred_w, pred_h, gt) if candidates else 0.0
                    oracle_iou = iou_box(cx, cy, gt.width, gt.height, gt) if candidates else 0.0
                    gt_records.append(
                        {
                            "sequence": sequence,
                            "proposal_hit": proposal_hit,
                            "ranker_hit": ranker_hit,
                            "centre_error": centre_err,
                            "iou": iou,
                            "oracle_iou": oracle_iou,
                        }
                    )
                    detail_by_config[config.label].append(
                        {
                            "fold": fold_id,
                            "sequence": sequence,
                            "proposal_hit": proposal_hit,
                            "ranker_hit": ranker_hit,
                            "centre_error": centre_err,
                            "iou": iou,
                            "oracle_iou": oracle_iou,
                            "iou50": float(iou >= 0.5),
                            "iou75": float(iou >= 0.75),
                        }
                    )

        metrics = summarize_gt_records(gt_records, window_times)
        metrics.update({"fold": fold_id, "config": config.label})
        fold_rows.append(metrics)
        for sequence in val_sequences:
            seq_gt = [r for r in gt_records if r["sequence"] == sequence]
            if not seq_gt:
                continue
            sm = summarize_gt_records(seq_gt, [])
            sm.update({"fold": fold_id, "config": config.label, "sequence": sequence})
            seq_rows[config.label].append(sm)

    flat_seq = [row for rows in seq_rows.values() for row in rows]
    return fold_rows, flat_seq, detail_by_config


def summarize_gt_records(records: list[dict], window_times: list[float]) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {"n_gt": 0.0}
    ious = np.asarray([float(r["iou"]) for r in records], dtype=np.float64)
    centre = np.asarray([float(r["centre_error"]) for r in records if not math.isnan(r["centre_error"])], dtype=np.float64)
    by_seq: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_seq[r["sequence"]].append(float(r["iou"] >= 0.5))
    out = {
        "n_gt": float(n),
        "proposal_contains_gt_pct": 100.0 * float(np.mean([r["proposal_hit"] for r in records])),
        "ranker_selected_gt_compatible_pct": 100.0 * float(np.mean([r["ranker_hit"] for r in records])),
        "centre_error_mean": float(np.mean(centre)) if len(centre) else float("nan"),
        "centre_error_median": float(np.median(centre)) if len(centre) else float("nan"),
        "centre_error_p90": float(np.percentile(centre, 90)) if len(centre) else float("nan"),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "pooled_micro_iou50_pct": 100.0 * float(np.mean(ious >= 0.5)),
        "pooled_micro_iou75_pct": 100.0 * float(np.mean(ious >= 0.75)),
        "sequence_macro_iou50_pct": 100.0 * float(np.mean([np.mean(v) for v in by_seq.values()])),
    }
    if window_times:
        wt = np.asarray(window_times, dtype=np.float64)
        out["inference_p50_ms"] = float(np.percentile(wt, 50))
        out["inference_p95_ms"] = float(np.percentile(wt, 95))
        out["inference_p99_ms"] = float(np.percentile(wt, 99))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end positive-window CV (deployable inference path)")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--table", default="artifacts/candidate_table.csv")
    parser.add_argument("--folds", default="sequence_folds.json")
    parser.add_argument("--out-dir", default="docs/runs/2026-08-30/end_to_end_positive")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = load_table(Path(args.table))
    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))

    main_configs = [
        EvalConfig(f"M1_{s}", "M1_logistic", "C1", s) for s in ("S0", "S1", "S2")
    ] + [EvalConfig(f"M2b_{s}", "M2b_extra_trees", "C1", s) for s in ("S0", "S1", "S2")]

    ablation_configs = [
        EvalConfig("A0", None, "C0", "S0"),
        EvalConfig("A1", "M2b_extra_trees", "C0", "S0"),
        EvalConfig("A2", "M2b_extra_trees", "C1", "S0"),
        EvalConfig("A3", "M2b_extra_trees", "C1", "S1"),
        EvalConfig("A4", "M2b_extra_trees", "C1", "S2"),
        EvalConfig("A2_M1", "M1_logistic", "C1", "S0"),
        EvalConfig("A4_M1", "M1_logistic", "C1", "S2"),
    ]

    all_fold_rows: list[dict] = []
    all_seq_rows: list[dict] = []
    all_details: dict[str, list[dict]] = defaultdict(list)

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    for fold in folds:
        fold_id = int(fold["fold"])
        train_sequences = set(fold["train"])
        val_sequences = set(fold["validation"])
        configs = main_configs + ablation_configs
        fold_rows, seq_rows, details = evaluate_fold(
            fold_id, train_sequences, val_sequences, table, split_dir, configs
        )
        all_fold_rows.extend(fold_rows)
        all_seq_rows.extend(seq_rows)
        for label, rows in details.items():
            all_details[label].extend(rows)
        write_csv("cv_by_fold_partial.csv", all_fold_rows)
        print(f"Fold {fold_id} complete ({len(val_sequences)} val sequences)")

    write_csv("cv_by_fold.csv", all_fold_rows)
    write_csv("cv_by_sequence.csv", all_seq_rows)

    ablation_labels = {c.label for c in ablation_configs}
    ablation_summary = []
    for label in sorted(ablation_labels):
        details = all_details.get(label, [])
        if not details:
            continue
        agg = aggregate_detection_metrics(details)
        agg["config"] = label
        fold_rows_for_label = [r for r in all_fold_rows if r["config"] == label]
        if fold_rows_for_label:
            for key in (
                "proposal_contains_gt_pct",
                "ranker_selected_gt_compatible_pct",
                "inference_p50_ms",
                "inference_p95_ms",
                "inference_p99_ms",
            ):
                vals = [float(r[key]) for r in fold_rows_for_label if key in r]
                if vals:
                    agg[key] = float(np.mean(vals))
        ablation_summary.append(agg)
    write_csv("ablation_summary.csv", ablation_summary)

    main_summary = []
    for label in [c.label for c in main_configs]:
        details = all_details.get(label, [])
        if not details:
            continue
        agg = aggregate_detection_metrics(details)
        agg["config"] = label
        main_summary.append(agg)
    write_csv("main_summary.csv", main_summary)

    best_label = max(ablation_summary, key=lambda r: float(r["pooled_micro_iou50_pct"]))["config"]
    best_details = all_details[best_label]
    bucket_rows = failure_buckets(best_details)
    write_csv("failure_buckets.csv", bucket_rows)

    md_path = Path("docs/runs/2026-08-30_end_to_end_positive.md")
    lines = [
        "# End-to-end positive-window CV — 2026-08-30",
        "",
        "Deployable path: deterministic Top-20 → ranker → C1 centre → size baseline. No GT at inference.",
        "",
        "## Main CV (ranker × size)",
        "",
    ]
    for label in [c.label for c in main_configs]:
        rows = [r for r in all_fold_rows if r["config"] == label]
        details = all_details.get(label, [])
        if not rows or not details:
            continue
        pooled = aggregate_detection_metrics(details)
        lines.append(f"### {label}")
        lines.append(
            f"- pooled micro IoU>=0.5: {pooled['pooled_micro_iou50_pct']:.3f}% "
            f"(n_gt={int(pooled['n_gt'])})"
        )
        lines.append(
            f"- sequence macro IoU>=0.5: {pooled['sequence_macro_iou50_pct']:.3f}% "
            f"fold mean IoU>=0.5: {pooled['fold_mean_iou50_pct']:.3f}%"
        )
        for r in rows:
            lines.append(
                f"- fold {int(r['fold'])}: proposal={r['proposal_contains_gt_pct']:.2f}% "
                f"ranker={r['ranker_selected_gt_compatible_pct']:.2f}% "
                f"pooled50={r['pooled_micro_iou50_pct']:.2f}% seq_macro50={r['sequence_macro_iou50_pct']:.2f}% "
                f"mean_iou={r['mean_iou']:.4f} infer_p95={r.get('inference_p95_ms', float('nan')):.3f}ms"
            )
        lines.append("")

    lines.extend(["## Ablation", ""])
    lines.append(
        "| config | pooled micro IoU>=0.5 % | sequence macro IoU>=0.5 % | fold mean IoU>=0.5 % | n_gt | mean IoU | ranker hit % |"
    )
    lines.append(
        "|--------|-------------------------:|---------------------------:|---------------------:|-----:|---------:|-------------:|"
    )
    for row in ablation_summary:
        lines.append(
            f"| {row['config']} | {row['pooled_micro_iou50_pct']:.3f} | {row['sequence_macro_iou50_pct']:.3f} | "
            f"{row['fold_mean_iou50_pct']:.3f} | {int(row['n_gt'])} | "
            f"{row['mean_iou']:.4f} | {row.get('ranker_selected_gt_compatible_pct', float('nan')):.2f} |"
        )

    lines.extend(["", f"## Failure buckets (config={best_label})", ""])
    for row in bucket_rows:
        lines.append(
            f"- {row['bucket']}: {row['count']} "
            f"({row['pct_all_gt']:.2f}% of all GT, {row['pct_failures_only']:.2f}% of failures)"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"best_ablation={best_label} md={md_path} out_dir={out_dir}")


if __name__ == "__main__":
    main()
