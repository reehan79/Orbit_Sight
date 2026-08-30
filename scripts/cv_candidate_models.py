from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from orbitsight.features import FEATURE_NAMES
from orbitsight.models import fit_bbox_ridge, fit_rankers, score_ranker
from orbitsight.proposals import infer_sensor_geometry

TOPKS = (1, 3, 5)


def load_table(path: Path) -> dict[str, np.ndarray]:
    sequences: list[str] = []
    sensors: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    ranks: list[int] = []
    targets: list[int] = []
    features: list[list[float]] = []
    bbox_targets: list[list[float]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sequences.append(row["sequence"])
            sensors.append(row["sensor"])
            starts.append(int(row["window_start_us"]))
            ends.append(int(row["window_end_us"]))
            ranks.append(int(row["candidate_rank"]))
            targets.append(int(row["target"]))
            features.append([float(row[name]) for name in FEATURE_NAMES])
            if row["bbox_dx_cells"] == "":
                bbox_targets.append([np.nan, np.nan, np.nan, np.nan])
            else:
                bbox_targets.append(
                    [
                        float(row["bbox_dx_cells"]),
                        float(row["bbox_dy_cells"]),
                        float(row["bbox_log_w_cells"]),
                        float(row["bbox_log_h_cells"]),
                    ]
                )

    return {
        "sequence": np.asarray(sequences, dtype=object),
        "sensor": np.asarray(sensors, dtype=object),
        "start": np.asarray(starts, dtype=np.int64),
        "end": np.asarray(ends, dtype=np.int64),
        "rank": np.asarray(ranks, dtype=np.int16),
        "target": np.asarray(targets, dtype=np.int8),
        "X": np.asarray(features, dtype=np.float32),
        "bbox": np.asarray(bbox_targets, dtype=np.float32),
    }


def group_indices(data: dict[str, np.ndarray], indices: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    if len(indices) == 0:
        return groups
    current: list[int] = []
    last_key = None
    for idx in indices:
        key = (data["sequence"][idx], int(data["start"][idx]), int(data["end"][idx]))
        if last_key is not None and key != last_key:
            groups.append(np.asarray(current, dtype=np.int64))
            current = []
        current.append(int(idx))
        last_key = key
    if current:
        groups.append(np.asarray(current, dtype=np.int64))
    return groups


def ranking_metrics(data: dict[str, np.ndarray], groups: list[np.ndarray], scores: np.ndarray, index_lookup: dict[int, int]) -> dict[str, float]:
    hits = {k: 0 for k in TOPKS}
    reciprocal_ranks: list[float] = []
    valid_groups = 0
    per_sequence_hits: dict[str, dict[int, list[int]]] = defaultdict(lambda: {k: [0, 0] for k in TOPKS})

    for group in groups:
        local_scores = np.asarray([scores[index_lookup[int(i)]] for i in group])
        order = group[np.argsort(local_scores)[::-1]]
        labels = data["target"][order]
        positive = np.flatnonzero(labels == 1)
        if len(positive) == 0:
            first_rank = None
            reciprocal_ranks.append(0.0)
        else:
            first_rank = int(positive[0]) + 1
            reciprocal_ranks.append(1.0 / first_rank)
        valid_groups += 1
        seq = str(data["sequence"][group[0]])
        for k in TOPKS:
            hit = int(first_rank is not None and first_rank <= k)
            hits[k] += hit
            per_sequence_hits[seq][k][0] += hit
            per_sequence_hits[seq][k][1] += 1

    result = {f"top{k}_micro": hits[k] / max(valid_groups, 1) for k in TOPKS}
    for k in TOPKS:
        seq_values = [h[k][0] / max(h[k][1], 1) for h in per_sequence_hits.values()]
        result[f"top{k}_macro"] = float(np.mean(seq_values)) if seq_values else float("nan")
    result["mrr"] = float(np.mean(reciprocal_ranks)) if valid_groups else 0.0
    result["groups"] = float(valid_groups)
    return result


def bbox_from_row(data: dict[str, np.ndarray], idx: int, target_values: np.ndarray) -> tuple[float, float, float, float]:
    sequence = str(data["sequence"][idx])
    width, height, cell = infer_sensor_geometry(sequence)
    feature_map = {name: float(data["X"][idx, j]) for j, name in enumerate(FEATURE_NAMES)}
    cx = feature_map["cx_normalized"] * width
    cy = feature_map["cy_normalized"] * height
    dx, dy, log_w, log_h = [float(v) for v in target_values]
    return (
        cx + dx * cell,
        cy + dy * cell,
        math.exp(log_w) * cell,
        math.exp(log_h) * cell,
    )


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def selected_bbox_metrics(
    data: dict[str, np.ndarray],
    groups: list[np.ndarray],
    scores: np.ndarray,
    index_lookup: dict[int, int],
    bbox_predictions: np.ndarray,
) -> dict[str, float]:
    values: list[float] = []
    selected_positive = 0
    evaluable = 0
    for group in groups:
        positives = group[data["target"][group] == 1]
        if len(positives) == 0:
            continue
        local_scores = np.asarray([scores[index_lookup[int(i)]] for i in group])
        selected = int(group[int(np.argmax(local_scores))])
        selected_local = index_lookup[selected]
        gt_boxes = []
        seen = set()
        for p in positives:
            box = bbox_from_row(data, int(p), data["bbox"][int(p)])
            key = tuple(round(v, 3) for v in box)
            if key not in seen:
                seen.add(key)
                gt_boxes.append(box)
        predicted = bbox_from_row(data, selected, bbox_predictions[selected_local])
        best = max(iou(predicted, gt) for gt in gt_boxes)
        values.append(best)
        selected_positive += int(data["target"][selected] == 1)
        evaluable += 1
    arr = np.asarray(values, dtype=np.float64)
    return {
        "bbox_groups": float(evaluable),
        "selected_positive_rate": selected_positive / max(evaluable, 1),
        "bbox_mean_iou": float(np.mean(arr)) if len(arr) else float("nan"),
        "bbox_median_iou": float(np.median(arr)) if len(arr) else float("nan"),
        "bbox_iou50": float(np.mean(arr >= 0.5)) if len(arr) else float("nan"),
    }


def benchmark_scorer(bundle, X: np.ndarray, ranks: np.ndarray, groups: list[np.ndarray], index_lookup: dict[int, int], limit: int = 500) -> tuple[float, float]:
    selected_groups = groups[:limit]
    if not selected_groups:
        return float("nan"), float("nan")

    first = np.asarray([index_lookup[int(i)] for i in selected_groups[0]], dtype=np.int64)
    # Warm prediction path before measuring. This avoids charging one-time sklearn
    # dispatch / allocation work to the first 40-ms observation.
    for _ in range(3):
        score_ranker(bundle, X[first], ranks[first])

    times = []
    for group in selected_groups:
        local = np.asarray([index_lookup[int(i)] for i in group], dtype=np.int64)
        t0 = time.perf_counter()
        score_ranker(bundle, X[local], ranks[local])
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return float(np.mean(arr)), float(np.percentile(arr, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-sequence CV for minimal OrbitSight candidate rankers")
    parser.add_argument("--table", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--out", default="artifacts/candidate_model_cv.csv")
    parser.add_argument(
        "--models",
        default="M0_raw_rank,M1_logistic,M2a_tree,M2b_extra_trees,M2_hist_gb",
        help="Comma-separated rankers. Use the fast subset to avoid refitting HGB.",
    )
    args = parser.parse_args()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    data = load_table(Path(args.table))
    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    print(f"rows={len(data['target'])} features={data['X'].shape[1]} positives={int(data['target'].sum())}")
    print(f"models={','.join(model_names)}")

    for fold in folds:
        fold_id = int(fold["fold"])
        train_sequences = set(fold["train"])
        validation_sequences = set(fold["validation"])
        train_idx = np.flatnonzero(np.isin(data["sequence"], list(train_sequences)))
        val_idx = np.flatnonzero(np.isin(data["sequence"], list(validation_sequences)))
        groups = group_indices(data, val_idx)
        lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(val_idx)}

        print(f"\nFold {fold_id}: train_rows={len(train_idx)} val_rows={len(val_idx)} val_groups={len(groups)}")
        rankers = fit_rankers(
            data["X"][train_idx],
            data["target"][train_idx],
            data["rank"][train_idx],
            model_names=model_names,
        )

        positive_train = train_idx[data["target"][train_idx] == 1]
        bbox_model = fit_bbox_ridge(data["X"][positive_train], data["bbox"][positive_train])
        bbox_pred = bbox_model.predict(data["X"][val_idx]).astype(np.float32)

        for name, bundle in rankers.items():
            scores = score_ranker(bundle, data["X"][val_idx], data["rank"][val_idx])
            metrics = ranking_metrics(data, groups, scores, lookup)
            bbox_metrics = selected_bbox_metrics(data, groups, scores, lookup, bbox_pred)
            score_mean_ms, score_p95_ms = benchmark_scorer(bundle, data["X"][val_idx], data["rank"][val_idx], groups, lookup)
            row: dict[str, object] = {"fold": fold_id, "model": name, **metrics, **bbox_metrics, "score_mean_ms": score_mean_ms, "score_p95_ms": score_p95_ms}
            rows.append(row)
            print(
                f"{name:16s} T1={100*metrics['top1_micro']:6.2f}% T3={100*metrics['top3_micro']:6.2f}% "
                f"T5={100*metrics['top5_micro']:6.2f}% macroT1={100*metrics['top1_macro']:6.2f}% "
                f"MRR={metrics['mrr']:.4f} bboxIoU50={100*bbox_metrics['bbox_iou50']:6.2f}% "
                f"score_p95={score_p95_ms:.4f} ms"
            )

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== CROSS-FOLD SUMMARY ===")
    for model in sorted({str(r["model"]) for r in rows}):
        selected = [r for r in rows if r["model"] == model]
        for metric in ("top1_micro", "top3_micro", "top5_micro", "top1_macro", "mrr", "bbox_iou50", "bbox_mean_iou", "score_p95_ms"):
            values = [float(r[metric]) for r in selected]
            print(f"{model:16s} {metric:18s} mean={np.mean(values):.6f} std={np.std(values):.6f}")
        print()
    print(f"csv={out_path}")
    print("NOTE: this experiment uses GT-positive windows only. Candidate-ranking and conditional box metrics are not official AP/F1/precision.")


if __name__ == "__main__":
    main()
