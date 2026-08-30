from __future__ import annotations

"""PART 0 — one-shot hybrid H1_NEURAL_SELECT_CLASSICAL_BOX vs B_CURRENT.

Uses already-trained fold ONNX models from artifacts/tiny_foveated_onnx.
No retraining of the neural net; ranker + S2 refit per fold (frozen recipe).
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from orbitsight.evaluation.detection_aggregate import aggregate_detection_metrics
from orbitsight.evaluation.gt_assignment import compatible
from orbitsight.features import (
    extract_candidate_features,
    extract_local_geometry_features,
    rasterize_event_patch,
    refine_c1_centroid,
    refine_c4_median,
)
from orbitsight.io import Detection, read_detection_file
from orbitsight.models import fit_rankers, score_ranker
from orbitsight.proposals import RawGridProposer, infer_sensor_geometry

PRIOR_MS = 80
TOP_K = 20
RANKER = "M2b_extra_trees"
SPLIT = Path(r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets")


def sensor_name(sequence: str) -> str:
    u = sequence.upper()
    if u.startswith("DAVIS"):
        return "DAVIS"
    if u.startswith("DVX"):
        return "DVX"
    return "EVK4"


def iou_box(cx, cy, w, h, gt: Detection) -> float:
    ax1, ay1 = cx - w / 2.0, cy - h / 2.0
    ax2, ay2 = cx + w / 2.0, cy + h / 2.0
    bx1, by1 = gt.cx - gt.width / 2.0, gt.cy - gt.height / 2.0
    bx2, by2 = gt.cx + gt.width / 2.0, gt.cy + gt.height / 2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w * h + gt.width * gt.height - inter
    return inter / union if union > 0 else 0.0


def load_table(path: Path) -> dict[str, np.ndarray]:
    from orbitsight.features import FEATURE_NAMES

    sequences, ranks, targets, features, bbox_targets, starts, ends = [], [], [], [], [], [], []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequences.append(row["sequence"])
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
        "start": np.asarray(starts, dtype=np.int64),
        "end": np.asarray(ends, dtype=np.int64),
        "rank": np.asarray(ranks, dtype=np.int16),
        "target": np.asarray(targets, dtype=np.int8),
        "X": np.asarray(features, dtype=np.float32),
        "bbox_log_wh": np.asarray(bbox_targets, dtype=np.float32),
    }


def fit_size_extratrees(X, y):
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=32, max_depth=12, min_samples_leaf=24,
        max_features=None, random_state=42, n_jobs=1,
    )
    model.fit(X, y)
    return model


def build_s2_training(table, train_idx, split_dir: Path):
    X_rows, y_rows = [], []
    pos_idx = train_idx[table["target"][train_idx] == 1]
    by_window: dict[tuple, list[int]] = defaultdict(list)
    for idx in pos_idx:
        by_window[(str(table["sequence"][idx]), int(table["start"][idx]), int(table["end"][idx]))].append(int(idx))
    for (sequence, start, end), indices in by_window.items():
        arr = np.load(split_dir / f"{sequence}_labeled_events.npy", mmap_mode="r")
        ts = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        left = int(np.searchsorted(ts, start, side="left"))
        right = int(np.searchsorted(ts, end, side="left"))
        current = np.asarray(arr[left:right, :4])
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        candidates = proposer.propose(current)
        if not candidates:
            continue
        for idx in indices:
            rank = int(table["rank"][idx])
            if rank < 1 or rank > len(candidates):
                continue
            cand = candidates[rank - 1]
            rcx, rcy = refine_c1_centroid(current, cand.cx, cand.cy, cell)
            local18 = extract_local_geometry_features(current, rcx, rcy, cell, width, height)
            X_rows.append(np.concatenate([table["X"][idx], local18]))
            y_rows.append(table["bbox_log_wh"][idx].tolist())
    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def onnx_infer(session, patches, features):
    outs = session.run(None, {"patch": patches.astype(np.float32), "features": features.astype(np.float32)})
    return outs[0], outs[1]


def classical_box(current, selected, feat15, cell, width, height, size_trees):
    cx, cy = refine_c4_median(current, selected.cx, selected.cy, cell)
    c1x, c1y = refine_c1_centroid(current, selected.cx, selected.cy, cell)
    local18 = extract_local_geometry_features(current, c1x, c1y, cell, width, height)
    log_wh = size_trees.predict(np.concatenate([feat15, local18]).reshape(1, -1))[0]
    return cx, cy, math.exp(float(log_wh[0])) * cell, math.exp(float(log_wh[1])) * cell


def evaluate_fold(fold_id, train_seqs, val_seqs, table, split_dir, onnx_path):
    import onnxruntime as ort

    train_idx = np.flatnonzero(np.isin(table["sequence"], list(train_seqs)))
    rankers = fit_rankers(table["X"][train_idx], table["target"][train_idx], table["rank"][train_idx], model_names=[RANKER])
    bundle = rankers[RANKER]
    size_X, size_y = build_s2_training(table, train_idx, split_dir)
    size_trees = fit_size_extratrees(size_X, size_y)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    details: dict[str, list[dict]] = defaultdict(list)
    for gt_path in sorted(split_dir.glob("*_bb_windows_40ms.txt")):
        sequence = gt_path.name.replace("_bb_windows_40ms.txt", "")
        if sequence not in val_seqs:
            continue
        sensor = sensor_name(sequence)
        arr = np.load(split_dir / f"{sequence}_labeled_events.npy", mmap_mode="r")
        ts = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        grouped: dict[tuple, list[Detection]] = defaultdict(list)
        for gt in read_detection_file(gt_path):
            grouped[(gt.start_us, gt.end_us)].append(gt)

        for (start_us, end_us), gts in sorted(grouped.items()):
            left = int(np.searchsorted(ts, start_us, side="left"))
            right = int(np.searchsorted(ts, end_us, side="left"))
            prior_left = int(np.searchsorted(ts, start_us - PRIOR_MS * 1000, side="left"))
            current = np.asarray(arr[left:right, :4])
            prior = np.asarray(arr[prior_left:left, :4])
            candidates = proposer.propose(current)
            if not candidates:
                for label in ("B_CURRENT", "H1_NEURAL_SELECT_CLASSICAL_BOX"):
                    for gt in gts:
                        details[label].append(_miss_row(fold_id, sequence, sensor, label))
                continue

            features = extract_candidate_features(current, prior, candidates, width, height, cell)
            ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
            scores = score_ranker(bundle, features, ranks)
            order = np.argsort(-scores)
            top1 = candidates[int(order[0])]
            top3 = [candidates[int(i)] for i in order[: min(3, len(candidates))]]

            # B_CURRENT
            b_cx, b_cy, b_w, b_h = classical_box(
                current, top1, features[candidates.index(top1)], cell, width, height, size_trees
            )

            # H1: neural cls on top3, classical box on winner
            patches = [rasterize_event_patch(current, prior, c.cx, c.cy, float(cell), start_us, end_us) for c in top3]
            feats3 = [features[candidates.index(c)].astype(np.float32) for c in top3]
            cls_logits, _ = onnx_infer(session, np.stack(patches), np.stack(feats3))
            pick = int(np.argmax(cls_logits))
            h1_sel = top3[pick]
            h1_cx, h1_cy, h1_w, h1_h = classical_box(
                current, h1_sel, features[candidates.index(h1_sel)], cell, width, height, size_trees
            )

            boxes = {
                "B_CURRENT": (b_cx, b_cy, b_w, b_h, top1),
                "H1_NEURAL_SELECT_CLASSICAL_BOX": (h1_cx, h1_cy, h1_w, h1_h, h1_sel),
            }
            for gt in gts:
                proposal_hit = any(compatible(c, gt, float(cell)) for c in candidates)
                for label, (cx, cy, w, h, sel) in boxes.items():
                    iou = iou_box(cx, cy, w, h, gt)
                    details[label].append(
                        {
                            "fold": fold_id,
                            "sequence": sequence,
                            "sensor": sensor,
                            "config": label,
                            "proposal_hit": proposal_hit,
                            "neural_selected_hit": compatible(sel, gt, float(cell)),
                            "centre_error": math.hypot(cx - gt.cx, cy - gt.cy),
                            "iou": iou,
                            "iou50": float(iou >= 0.5),
                            "iou75": float(iou >= 0.75),
                        }
                    )
    return details


def _miss_row(fold_id, sequence, sensor, label):
    return {
        "fold": fold_id,
        "sequence": sequence,
        "sensor": sensor,
        "config": label,
        "proposal_hit": False,
        "neural_selected_hit": False,
        "centre_error": float("nan"),
        "iou": 0.0,
        "iou50": 0.0,
        "iou75": 0.0,
    }


def summarize(details):
    agg = aggregate_detection_metrics(details)
    n = len(details)
    if n:
        agg["neural_selected_compatible_pct"] = 100.0 * sum(float(d["neural_selected_hit"]) for d in details) / n
    return agg


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=SPLIT)
    parser.add_argument("--table", type=Path, default=Path("artifacts/candidate_table.csv"))
    parser.add_argument("--folds", type=Path, default=Path("sequence_folds.json"))
    parser.add_argument("--cross-sensor-folds", type=Path, default=Path("docs/runs/2026-08-30/cross_sensor_folds.json"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/tiny_foveated_onnx"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/runs/2026-08-30/challenge_metric_baseline"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = load_table(args.table)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    all_details: dict[str, list] = defaultdict(list)

    for fold in folds:
        fold_id = int(fold["fold"])
        print(f"H1 fold {fold_id}...", flush=True)
        onnx_path = args.onnx_dir / f"fold{fold_id}_refiner.onnx"
        details = evaluate_fold(fold_id, set(fold["train"]), set(fold["validation"]), table, args.split_dir, onnx_path)
        for k, v in details.items():
            all_details[k].extend(v)

    summary = []
    for label in ("B_CURRENT", "H1_NEURAL_SELECT_CLASSICAL_BOX"):
        agg = summarize(all_details[label])
        agg["config"] = label
        summary.append(agg)
    write_csv(args.out_dir / "h1_summary.csv", summary)

    by_sensor, by_seq = [], []
    for label, rows in all_details.items():
        groups_s: dict[str, list] = defaultdict(list)
        groups_q: dict[str, list] = defaultdict(list)
        for r in rows:
            groups_s[r["sensor"]].append(r)
            groups_q[r["sequence"]].append(r)
        for g, rs in sorted(groups_s.items()):
            a = summarize(rs)
            a.update({"config": label, "sensor": g})
            by_sensor.append(a)
        for g, rs in sorted(groups_q.items()):
            a = summarize(rs)
            a.update({"config": label, "sequence": g})
            by_seq.append(a)
    write_csv(args.out_dir / "h1_by_sensor.csv", by_sensor)
    write_csv(args.out_dir / "h1_by_sequence.csv", by_seq)

    # Cross-sensor
    cross_folds = json.loads(args.cross_sensor_folds.read_text(encoding="utf-8"))
    cross_details: dict[str, list] = defaultdict(list)
    for fold in cross_folds:
        fold_id = int(fold["fold"])
        print(f"H1 cross-sensor fold {fold_id}...", flush=True)
        onnx_path = args.onnx_dir / "cross" / f"fold{fold_id}_refiner.onnx"
        details = evaluate_fold(fold_id, set(fold["train"]), set(fold["validation"]), table, args.split_dir, onnx_path)
        for label in ("B_CURRENT", "H1_NEURAL_SELECT_CLASSICAL_BOX"):
            cross_details[label].extend(details[label])
    cross_summary = []
    for label in ("B_CURRENT", "H1_NEURAL_SELECT_CLASSICAL_BOX"):
        agg = summarize(cross_details[label])
        agg["config"] = label
        cross_summary.append(agg)
    write_csv(args.out_dir / "h1_cross_sensor.csv", cross_summary)

    b = next(r for r in summary if r["config"] == "B_CURRENT")
    h = next(r for r in summary if r["config"] == "H1_NEURAL_SELECT_CLASSICAL_BOX")
    print(
        f"B pooled={b['pooled_micro_iou50_pct']:.3f} seq={b['sequence_macro_iou50_pct']:.3f} "
        f"H1 pooled={h['pooled_micro_iou50_pct']:.3f} seq={h['sequence_macro_iou50_pct']:.3f}",
        flush=True,
    )
    beat = (
        float(h["pooled_micro_iou50_pct"]) > float(b["pooled_micro_iou50_pct"])
        and float(h["sequence_macro_iou50_pct"]) > float(b["sequence_macro_iou50_pct"])
    )
    print(f"H1_beats_B_CURRENT={beat}", flush=True)
    (args.out_dir / "h1_gate.txt").write_text(f"H1_beats_B_CURRENT={beat}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
