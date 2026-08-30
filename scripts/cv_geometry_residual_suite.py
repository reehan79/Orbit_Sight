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

from orbitsight.evaluation.detection_aggregate import aggregate_detection_metrics
from orbitsight.features import (
    FEATURE_NAMES,
    extract_candidate_features,
    extract_local_geometry_features,
    local_extent_from_roi,
    refine_c1_centroid,
    refine_c4_median,
    refine_c5_soft_background_centroid,
)
from orbitsight.io import Detection, read_detection_file
from orbitsight.models import fit_rankers, score_ranker
from orbitsight.proposals import Candidate, RawGridProposer, infer_sensor_geometry

PRIOR_MS = 80
TOP_K = 20
RANKER = "M2b_extra_trees"

CENTRE_METHODS = (
    "C1_CENTROID",
    "C4_MEDIAN",
    "C5_SOFT_BACKGROUND_CENTROID",
    "C6_RIDGE_RESIDUAL",
    "C7_EXTRATREES_RESIDUAL",
)
SIZE_METHODS = ("S2", "S3", "S4")
ORACLE_MODES = ("ACTUAL", "ORACLE_SIZE", "ORACLE_CENTRE", "ORACLE_BOTH")


@dataclass(frozen=True)
class SuiteConfig:
    label: str
    centre: str
    size: str


def config_label(centre: str, size: str) -> str:
    return f"{centre}__{size}"


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


def fit_ridge(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([("scale", StandardScaler()), ("regressor", Ridge(alpha=2.0))])
    model.fit(X, y)
    return model


def build_s2_training(
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


def build_centre_residual_training(
    split_dir: Path,
    train_sequences: set[str],
    ranker_bundle,
) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y_rows: list[list[float]] = []
    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        if sequence not in train_sequences:
            continue
        npy_path = split_dir / f"{sequence}_labeled_events.npy"
        arr = np.load(npy_path, mmap_mode="r")
        timestamps = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
        for gt in read_detection_file(gt_path):
            grouped[(gt.start_us, gt.end_us)].append(gt)
        for (start_us, end_us), gts in grouped.items():
            left = int(np.searchsorted(timestamps, start_us, side="left"))
            right = int(np.searchsorted(timestamps, end_us, side="left"))
            prior_left = int(np.searchsorted(timestamps, start_us - PRIOR_MS * 1000, side="left"))
            current = np.asarray(arr[left:right, :4])
            prior = np.asarray(arr[prior_left:left, :4])
            candidates = proposer.propose(current)
            if not candidates:
                continue
            features = extract_candidate_features(current, prior, candidates, width, height, cell)
            ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
            scores = score_ranker(ranker_bundle, features, ranks)
            selected = candidates[int(np.argmax(scores))]
            sel_idx = candidates.index(selected)
            feat15 = features[sel_idx]
            raw_cx, raw_cy = selected.cx, selected.cy
            c1_cx, c1_cy = refine_c1_centroid(current, raw_cx, raw_cy, cell)
            local18 = extract_local_geometry_features(current, raw_cx, raw_cy, cell, width, height)
            c1_dx = (c1_cx - raw_cx) / cell
            c1_dy = (c1_cy - raw_cy) / cell
            feat = np.concatenate([feat15, local18, np.array([c1_dx, c1_dy], dtype=np.float32)])
            for gt in gts:
                if not compatible(selected, gt, float(cell)):
                    continue
                y_rows.append([(gt.cx - c1_cx) / cell, (gt.cy - c1_cy) / cell])
                X_rows.append(feat)
    if not X_rows:
        return np.empty((0, 35), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def build_s4_training(
    split_dir: Path,
    train_sequences: set[str],
    ranker_bundle,
) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y_rows: list[list[float]] = []
    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        if sequence not in train_sequences:
            continue
        npy_path = split_dir / f"{sequence}_labeled_events.npy"
        arr = np.load(npy_path, mmap_mode="r")
        timestamps = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        grouped: dict[tuple[int, int], list[Detection]] = defaultdict(list)
        for gt in read_detection_file(gt_path):
            grouped[(gt.start_us, gt.end_us)].append(gt)
        for (start_us, end_us), gts in grouped.items():
            left = int(np.searchsorted(timestamps, start_us, side="left"))
            right = int(np.searchsorted(timestamps, end_us, side="left"))
            prior_left = int(np.searchsorted(timestamps, start_us - PRIOR_MS * 1000, side="left"))
            current = np.asarray(arr[left:right, :4])
            prior = np.asarray(arr[prior_left:left, :4])
            candidates = proposer.propose(current)
            if not candidates:
                continue
            features = extract_candidate_features(current, prior, candidates, width, height, cell)
            ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
            scores = score_ranker(ranker_bundle, features, ranks)
            selected = candidates[int(np.argmax(scores))]
            sel_idx = candidates.index(selected)
            feat15 = features[sel_idx]
            cx, cy = refine_c1_centroid(current, selected.cx, selected.cy, cell)
            s3_w, s3_h = local_extent_from_roi(current, cx, cy, cell)
            local18 = extract_local_geometry_features(current, cx, cy, cell, width, height)
            feat = np.concatenate(
                [feat15, local18, np.array([math.log(max(s3_w / cell, 1e-6)), math.log(max(s3_h / cell, 1e-6))], dtype=np.float32)]
            )
            for gt in gts:
                if not compatible(selected, gt, float(cell)):
                    continue
                y_rows.append([math.log(gt.width / max(s3_w, 1e-6)), math.log(gt.height / max(s3_h, 1e-6))])
                X_rows.append(feat)
    if not X_rows:
        return np.empty((0, 35), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def refine_centre(
    centre_mode: str,
    current: np.ndarray,
    prior: np.ndarray,
    candidate: Candidate,
    cell: float,
    width: int,
    height: int,
    feat15: np.ndarray,
    centre_ridge,
    centre_trees,
) -> tuple[float, float]:
    raw_cx, raw_cy = candidate.cx, candidate.cy
    c1_cx, c1_cy = refine_c1_centroid(current, raw_cx, raw_cy, cell)
    if centre_mode == "C1_CENTROID":
        return c1_cx, c1_cy
    if centre_mode == "C4_MEDIAN":
        return refine_c4_median(current, raw_cx, raw_cy, cell)
    if centre_mode == "C5_SOFT_BACKGROUND_CENTROID":
        return refine_c5_soft_background_centroid(current, prior, raw_cx, raw_cy, cell)
    if centre_mode in ("C6_RIDGE_RESIDUAL", "C7_EXTRATREES_RESIDUAL"):
        local18 = extract_local_geometry_features(current, raw_cx, raw_cy, cell, width, height)
        c1_dx = (c1_cx - raw_cx) / cell
        c1_dy = (c1_cy - raw_cy) / cell
        feat = np.concatenate([feat15, local18, np.array([c1_dx, c1_dy], dtype=np.float32)])
        model = centre_ridge if centre_mode == "C6_RIDGE_RESIDUAL" else centre_trees
        residual = model.predict(feat.reshape(1, -1))[0]
        return c1_cx + float(residual[0]) * cell, c1_cy + float(residual[1]) * cell
    raise ValueError(centre_mode)


def predict_size(
    size_mode: str,
    cell: float,
    cx: float,
    cy: float,
    current: np.ndarray,
    feat15: np.ndarray,
    width: int,
    height: int,
    size_trees,
    size_extent_trees,
) -> tuple[float, float]:
    if size_mode == "S3":
        return local_extent_from_roi(current, cx, cy, cell)
    if size_mode == "S2":
        local18 = extract_local_geometry_features(current, cx, cy, cell, width, height)
        size_X = np.concatenate([feat15, local18])
        log_wh = size_trees.predict(size_X.reshape(1, -1))[0]
        return math.exp(float(log_wh[0])) * cell, math.exp(float(log_wh[1])) * cell
    if size_mode == "S4":
        s3_w, s3_h = local_extent_from_roi(current, cx, cy, cell)
        local18 = extract_local_geometry_features(current, cx, cy, cell, width, height)
        feat = np.concatenate(
            [feat15, local18, np.array([math.log(max(s3_w / cell, 1e-6)), math.log(max(s3_h / cell, 1e-6))], dtype=np.float32)]
        )
        residual = size_extent_trees.predict(feat.reshape(1, -1))[0]
        return s3_w * math.exp(float(residual[0])), s3_h * math.exp(float(residual[1]))
    raise ValueError(size_mode)


def evaluate_fold(
    fold_id: int,
    train_sequences: set[str],
    val_sequences: set[str],
    table: dict[str, np.ndarray],
    split_dir: Path,
    configs: list[SuiteConfig],
    collect_oracle: bool = False,
) -> tuple[dict[str, list[dict]], list[dict], list[dict]]:
    train_idx = np.flatnonzero(np.isin(table["sequence"], list(train_sequences)))
    rankers = fit_rankers(
        table["X"][train_idx],
        table["target"][train_idx],
        table["rank"][train_idx],
        model_names=[RANKER],
    )
    bundle = rankers[RANKER]
    size_X_train, size_y_train = build_s2_training(table, train_idx, split_dir)
    size_trees = fit_size_extratrees(size_X_train, size_y_train)
    centre_X, centre_y = build_centre_residual_training(split_dir, train_sequences, bundle)
    centre_ridge = fit_ridge(centre_X, centre_y) if len(centre_X) else None
    centre_trees = fit_size_extratrees(centre_X, centre_y) if len(centre_X) else None
    s4_X, s4_y = build_s4_training(split_dir, train_sequences, bundle)
    size_extent_trees = fit_size_extratrees(s4_X, s4_y) if len(s4_X) else None

    details_by_config: dict[str, list[dict]] = {c.label: [] for c in configs}
    oracle_rows: list[dict] = []
    fold_rows: list[dict] = []

    for config in configs:
        window_times: list[float] = []
        for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
            sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
            if sequence not in val_sequences:
                continue
            sensor = sensor_name(sequence)
            npy_path = split_dir / f"{sequence}_labeled_events.npy"
            arr = np.load(npy_path, mmap_mode="r")
            timestamps = arr[:, 3]
            width, height, cell = infer_sensor_geometry(sequence)
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
                if not candidates:
                    cx = cy = 0.0
                    pred_w = pred_h = 0.0
                    selected = None
                    feat15 = None
                else:
                    features = extract_candidate_features(current, prior, candidates, width, height, cell)
                    ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
                    scores = score_ranker(bundle, features, ranks)
                    selected = candidates[int(np.argmax(scores))]
                    sel_idx = candidates.index(selected)
                    feat15 = features[sel_idx]
                    cx, cy = refine_centre(
                        config.centre,
                        current,
                        prior,
                        selected,
                        cell,
                        width,
                        height,
                        feat15,
                        centre_ridge,
                        centre_trees,
                    )
                    pred_w, pred_h = predict_size(
                        config.size,
                        cell,
                        cx,
                        cy,
                        current,
                        feat15,
                        width,
                        height,
                        size_trees,
                        size_extent_trees,
                    )
                infer_ms = (perf_counter_ns() - t0) / 1_000_000.0
                window_times.append(infer_ms)

                for gt in gts:
                    if candidates:
                        proposal_hit = any(compatible(c, gt, float(cell)) for c in candidates)
                        ranker_hit = selected is not None and compatible(selected, gt, float(cell))
                        centre_err = math.hypot(cx - gt.cx, cy - gt.cy)
                        iou = iou_box(cx, cy, pred_w, pred_h, gt)
                        oracle_iou = iou_box(cx, cy, gt.width, gt.height, gt)
                    else:
                        proposal_hit = False
                        ranker_hit = False
                        centre_err = float("nan")
                        iou = 0.0
                        oracle_iou = 0.0

                    details_by_config[config.label].append(
                        {
                            "fold": fold_id,
                            "sequence": sequence,
                            "sensor": sensor,
                            "config": config.label,
                            "centre": config.centre,
                            "size": config.size,
                            "proposal_hit": proposal_hit,
                            "ranker_hit": ranker_hit,
                            "centre_error": centre_err,
                            "iou": iou,
                            "oracle_iou": oracle_iou,
                            "iou50": float(iou >= 0.5),
                            "iou75": float(iou >= 0.75),
                        }
                    )

                    if collect_oracle and config.centre == "C1_CENTROID" and config.size == "S2":
                        if ranker_hit:
                            modes = {
                                "ACTUAL": (cx, cy, pred_w, pred_h),
                                "ORACLE_SIZE": (cx, cy, gt.width, gt.height),
                                "ORACLE_CENTRE": (gt.cx, gt.cy, pred_w, pred_h),
                                "ORACLE_BOTH": (gt.cx, gt.cy, gt.width, gt.height),
                            }
                            for mode, (mcx, mcy, mw, mh) in modes.items():
                                miou = iou_box(mcx, mcy, mw, mh, gt)
                                oracle_rows.append(
                                    {
                                        "fold": fold_id,
                                        "sequence": sequence,
                                        "sensor": sensor,
                                        "mode": mode,
                                        "iou": miou,
                                        "iou50": float(miou >= 0.5),
                                    }
                                )
                        else:
                            for mode in ORACLE_MODES:
                                oracle_rows.append(
                                    {
                                        "fold": fold_id,
                                        "sequence": sequence,
                                        "sensor": sensor,
                                        "mode": mode,
                                        "iou": 0.0,
                                        "iou50": 0.0,
                                    }
                                )

        if window_times:
            wt = np.asarray(window_times, dtype=np.float64)
            fold_rows.append(
                {
                    "fold": fold_id,
                    "config": config.label,
                    "inference_p50_ms": float(np.percentile(wt, 50)),
                    "inference_p95_ms": float(np.percentile(wt, 95)),
                    "inference_p99_ms": float(np.percentile(wt, 99)),
                }
            )

    return details_by_config, fold_rows, oracle_rows


def summarize_by_key(details: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        groups[str(row[key])].append(row)
    out: list[dict] = []
    for group_key, rows in sorted(groups.items()):
        agg = aggregate_detection_metrics(rows)
        agg[key] = group_key
        agg["config"] = rows[0].get("config", "")
        out.append(agg)
    return out


def summarize_oracle(oracle_rows: list[dict]) -> list[dict]:
    out: list[dict] = []

    def _append(scope: str, group: str, mode: str, rows: list[dict]) -> None:
        ious = [float(r["iou"]) for r in rows]
        out.append(
            {
                "scope": scope,
                "group": group,
                "mode": mode,
                "n_gt": len(rows),
                "pooled_micro_iou50_pct": 100.0 * sum(float(r["iou50"]) for r in rows) / len(rows),
                "mean_iou": float(np.mean(ious)),
            }
        )

    by_mode: dict[str, list[dict]] = defaultdict(list)
    by_sensor_mode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_seq_mode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in oracle_rows:
        mode = str(row["mode"])
        by_mode[mode].append(row)
        by_sensor_mode[(str(row["sensor"]), mode)].append(row)
        by_seq_mode[(str(row["sequence"]), mode)].append(row)

    for mode, rows in sorted(by_mode.items()):
        _append("pooled", mode, mode, rows)
    for (sensor, mode), rows in sorted(by_sensor_mode.items()):
        _append("sensor", sensor, mode, rows)
    for (sequence, mode), rows in sorted(by_seq_mode.items()):
        _append("sequence", sequence, mode, rows)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_latency_summary(summary_rows: list[dict], details: list[dict], fold_rows: list[dict]) -> None:
    for row in summary_rows:
        cfg = row["config"]
        cfg_folds = [f for f in fold_rows if f.get("config") == cfg]
        if cfg_folds:
            for key in ("inference_p50_ms", "inference_p95_ms", "inference_p99_ms"):
                vals = [float(f[key]) for f in cfg_folds if key in f]
                if vals:
                    row[key] = float(np.mean(vals))


def run_cv(
    split_dir: Path,
    table: dict[str, np.ndarray],
    folds: list[dict],
    configs: list[SuiteConfig],
    collect_oracle: bool,
) -> tuple[dict[str, list[dict]], list[dict], list[dict]]:
    all_details: dict[str, list[dict]] = defaultdict(list)
    all_oracle: list[dict] = []
    fold_latency: list[dict] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        train_sequences = set(fold["train"])
        val_sequences = set(fold["validation"])
        details_by_config, fold_latency_rows, oracle_rows = evaluate_fold(
            fold_id, train_sequences, val_sequences, table, split_dir, configs, collect_oracle
        )
        for label, rows in details_by_config.items():
            all_details[label].extend(rows)
        all_oracle.extend(oracle_rows)
        fold_latency.extend(fold_latency_rows)

    return all_details, all_oracle, fold_latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry residual suite (centre × size diagnostics)")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--table", default="artifacts/candidate_table.csv")
    parser.add_argument("--folds", default="sequence_folds.json")
    parser.add_argument("--cross-sensor-folds", default="docs/runs/2026-08-30/cross_sensor_folds.json")
    parser.add_argument("--out-dir", default="docs/runs/2026-08-30/geometry_residual")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = load_table(Path(args.table))

    part2_configs = [SuiteConfig(config_label(c, "S2"), c, "S2") for c in CENTRE_METHODS]
    part3_configs = [
        SuiteConfig(config_label(c, s), c, s) for c in CENTRE_METHODS for s in SIZE_METHODS
    ]
    all_configs = {c.label: c for c in part3_configs}
    configs = list(all_configs.values())

    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    all_details, oracle_rows, fold_rows = run_cv(split_dir, table, folds, configs, collect_oracle=True)

    summary_rows: list[dict] = []
    for label in sorted(all_details):
        cfg = all_configs[label]
        details = all_details[label]
        agg = aggregate_detection_metrics(details)
        agg["config"] = label
        agg["centre"] = cfg.centre
        agg["size"] = cfg.size
        summary_rows.append(agg)
    add_latency_summary(summary_rows, all_details, fold_rows)
    write_csv(out_dir / "matrix_summary.csv", summary_rows)

    part2_summary = [r for r in summary_rows if r["size"] == "S2"]
    write_csv(out_dir / "centre_methods_s2.csv", part2_summary)

    by_sensor: list[dict] = []
    by_sequence: list[dict] = []
    for label, details in all_details.items():
        for row in summarize_by_key(details, "sensor"):
            row["config"] = label
            by_sensor.append(row)
        for row in summarize_by_key(details, "sequence"):
            row["config"] = label
            by_sequence.append(row)
    write_csv(out_dir / "by_sensor.csv", by_sensor)
    write_csv(out_dir / "by_sequence.csv", by_sequence)
    write_csv(out_dir / "cv_by_fold.csv", fold_rows)

    oracle_summary = summarize_oracle(oracle_rows)
    write_csv(out_dir / "oracle_decomposition.csv", oracle_summary)

    best_configs = sorted(summary_rows, key=lambda r: float(r["pooled_micro_iou50_pct"]), reverse=True)[:3]
    cross_configs = [all_configs[r["config"]] for r in best_configs]
    cross_folds = json.loads(Path(args.cross_sensor_folds).read_text(encoding="utf-8"))
    cross_details, _, cross_fold_rows = run_cv(split_dir, table, cross_folds, cross_configs, collect_oracle=False)
    cross_summary: list[dict] = []
    for label, details in cross_details.items():
        agg = aggregate_detection_metrics(details)
        agg["config"] = label
        cross_summary.append(agg)
    write_csv(out_dir / "cross_sensor_summary.csv", cross_summary)
    write_csv(out_dir / "cross_sensor_by_fold.csv", cross_fold_rows)

    md_path = Path("docs/runs/2026-08-30_geometry_residual_suite.md")
    lines = [
        "# Geometry residual suite — 2026-08-30",
        "",
        "Frozen M2b ranker. Centre methods C1/C4/C5/C6/C7 × size baselines S2/S3/S4.",
        "Metrics: **pooled micro** = sum(success)/sum(GT); **sequence macro** = unweighted mean of per-sequence IoU>=0.5; **fold mean** = unweighted mean of fold percentages.",
        "",
        "## Part 2 — Centre methods (S2 size)",
        "",
        "| centre | pooled micro IoU>=0.5 % | sequence macro % | fold mean % | n_gt | mean IoU | centre err mean | p95 ms |",
        "|--------|-------------------------:|-----------------:|------------:|-----:|---------:|----------------:|-------:|",
    ]
    for row in part2_summary:
        lines.append(
            f"| {row['centre']} | {row['pooled_micro_iou50_pct']:.3f} | {row['sequence_macro_iou50_pct']:.3f} | "
            f"{row['fold_mean_iou50_pct']:.3f} | {int(row['n_gt'])} | {row['mean_iou']:.4f} | "
            f"{row.get('centre_error_mean', float('nan')):.3f} | {row.get('inference_p95_ms', float('nan')):.3f} |"
        )

    lines.extend(["", "## Part 3 — Centre × size matrix", ""])
    lines.append(
        "| config | pooled micro IoU>=0.5 % | sequence macro % | fold mean % | IoU>=0.75 % | median IoU |"
    )
    lines.append("|--------|-------------------------:|-----------------:|------------:|------------:|-----------:|")
    for row in summary_rows:
        lines.append(
            f"| {row['config']} | {row['pooled_micro_iou50_pct']:.3f} | {row['sequence_macro_iou50_pct']:.3f} | "
            f"{row['fold_mean_iou50_pct']:.3f} | {row['pooled_micro_iou75_pct']:.3f} | {row['median_iou']:.4f} |"
        )

    lines.extend(["", "## Part 1 — Oracle bottleneck (A4 path: C1 + S2)", ""])
    for row in oracle_summary:
        if row["scope"] == "pooled":
            lines.append(
                f"- {row['group']}: pooled micro IoU>=0.5={row['pooled_micro_iou50_pct']:.3f}% "
                f"mean IoU={row['mean_iou']:.4f} (n_gt={int(row['n_gt'])})"
            )

    lines.extend(["", "## Part 4 — Cross-sensor stress (top matrix configs)", ""])
    for row in cross_summary:
        lines.append(
            f"- {row['config']}: pooled micro IoU>=0.5={row['pooled_micro_iou50_pct']:.3f}% "
            f"sequence macro={row['sequence_macro_iou50_pct']:.3f}% n_gt={int(row['n_gt'])}"
        )

    lines.extend(["", f"CSVs: `{out_dir}/`", ""])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report={md_path} out_dir={out_dir}")


if __name__ == "__main__":
    main()
