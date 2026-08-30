from __future__ import annotations

"""PARTS 5–10 — all-window confidence detector CV (challenge metrics)."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold

from orbitsight.evaluation.gt_assignment import compatible
from orbitsight.evaluation.tii_style import (
    TIIMetrics,
    aggregate_tii,
    evaluate_dirs_tii,
    load_tii_gt,
    match_and_score,
    write_tii_prediction_file,
)
from orbitsight.features.candidate_features_fast import extract_candidate_features_fast
from orbitsight.features import (
    FEATURE_NAMES,
    extract_local_geometry_features,
    refine_c1_centroid,
    refine_c4_median,
    rasterize_event_patch,
)
from orbitsight.inference.b_current import SequenceStream
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows
from orbitsight.io import Detection, read_detection_file
from orbitsight.models import fit_rankers, score_ranker
from orbitsight.proposals import RawGridProposer, infer_sensor_geometry

PRIOR_MS = 80
TOP_K = 20
RANKER = "M2b_extra_trees"
NMS_IOU = 0.30
SPLIT = Path(r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets")
TII_EVAL = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_DataLoader\evaluate.py"
)


def sensor_name(sequence: str) -> str:
    u = sequence.upper()
    if u.startswith("DAVIS"):
        return "DAVIS"
    if u.startswith("DVX"):
        return "DVX"
    return "EVK4"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_table(path: Path) -> dict[str, np.ndarray]:
    sequences, ranks, targets, features, bbox, starts, ends = [], [], [], [], [], [], []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequences.append(row["sequence"])
            starts.append(int(row["window_start_us"]))
            ends.append(int(row["window_end_us"]))
            ranks.append(int(row["candidate_rank"]))
            targets.append(int(row["target"]))
            features.append([float(row[n]) for n in FEATURE_NAMES])
            if row["bbox_log_w_cells"] == "":
                bbox.append([np.nan, np.nan])
            else:
                bbox.append([float(row["bbox_log_w_cells"]), float(row["bbox_log_h_cells"])])
    return {
        "sequence": np.asarray(sequences, object),
        "start": np.asarray(starts, np.int64),
        "end": np.asarray(ends, np.int64),
        "rank": np.asarray(ranks, np.int16),
        "target": np.asarray(targets, np.int8),
        "X": np.asarray(features, np.float32),
        "bbox_log_wh": np.asarray(bbox, np.float32),
    }


def fit_size_s2(table, train_idx, split_dir: Path):
    X_rows, y_rows = [], []
    pos = train_idx[table["target"][train_idx] == 1]
    by_w = defaultdict(list)
    for idx in pos:
        by_w[(str(table["sequence"][idx]), int(table["start"][idx]), int(table["end"][idx]))].append(int(idx))
    for (sequence, start, end), indices in by_w.items():
        arr = np.load(split_dir / f"{sequence}_labeled_events.npy", mmap_mode="r")
        ts = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        left = int(np.searchsorted(ts, start, side="left"))
        right = int(np.searchsorted(ts, end, side="left"))
        current = np.asarray(arr[left:right, :4])
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        cands = proposer.propose(current)
        if not cands:
            continue
        for idx in indices:
            rank = int(table["rank"][idx])
            if rank < 1 or rank > len(cands):
                continue
            cand = cands[rank - 1]
            rcx, rcy = refine_c1_centroid(current, cand.cx, cand.cy, cell)
            local18 = extract_local_geometry_features(current, rcx, rcy, cell, width, height)
            X_rows.append(np.concatenate([table["X"][idx], local18]))
            y_rows.append(table["bbox_log_wh"][idx].tolist())
    model = ExtraTreesRegressor(
        n_estimators=32, max_depth=12, min_samples_leaf=24,
        max_features=None, random_state=42, n_jobs=1,
    )
    model.fit(np.asarray(X_rows, np.float32), np.asarray(y_rows, np.float32))
    return model


def fit_confidence(X: np.ndarray, y: np.ndarray) -> ExtraTreesClassifier:
    clf = ExtraTreesClassifier(
        n_estimators=64,
        max_depth=14,
        min_samples_leaf=12,
        max_features=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X, y)
    return clf


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose threshold maximizing F1 on TRAIN-OOF scores only."""
    if len(y_true) == 0:
        return 0.5
    # Candidate thresholds from unique score quantiles
    qs = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_t, best_f1 = 0.5, -1.0
    for t in qs:
        pred = scores >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum((~pred) & (y_true == 1)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def oof_threshold_on_train(cache: dict, train_sequences: list[str]) -> float:
    """Sequence-level inner folds on OUTER TRAIN only."""
    mask = np.array([str(s) in set(train_sequences) for s in cache["sequence"]], dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0.5
    seqs = np.unique(cache["sequence"][idx].astype(str))
    if len(seqs) < 2:
        X = cache["features"][idx]
        y = cache["target"][idx]
        clf = fit_confidence(X, y)
        scores = clf.predict_proba(X)[:, 1]
        return select_f1_threshold(y, scores)

    n_splits = min(5, len(seqs))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_scores = np.zeros(len(idx), dtype=np.float64)
    oof_y = cache["target"][idx].astype(np.int8)
    seq_of = cache["sequence"][idx].astype(str)
    for train_s, val_s in kf.split(seqs):
        train_set = set(seqs[train_s])
        val_set = set(seqs[val_s])
        tr = np.array([s in train_set for s in seq_of])
        va = np.array([s in val_set for s in seq_of])
        if not tr.any() or not va.any():
            continue
        clf = fit_confidence(cache["features"][idx[tr]], cache["target"][idx[tr]])
        oof_scores[va] = clf.predict_proba(cache["features"][idx[va]])[:, 1]
    return select_f1_threshold(oof_y, oof_scores)


def box_iou(a, b) -> float:
    # continuous centre boxes for NMS
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def greedy_nms(dets: list[tuple], iou_thresh: float) -> list[tuple]:
    """dets: (cx, cy, w, h, conf, candidate). Keep high conf first."""
    order = sorted(range(len(dets)), key=lambda i: dets[i][4], reverse=True)
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(dets[i])
        for j in order:
            if j == i or j in suppressed:
                continue
            if box_iou(dets[i][:4], dets[j][:4]) >= iou_thresh:
                suppressed.add(j)
    return keep


def classical_box_from_candidate(current, cand, feat15, cell, width, height, size_trees):
    cx, cy = refine_c4_median(current, cand.cx, cand.cy, cell)
    c1x, c1y = refine_c1_centroid(current, cand.cx, cand.cy, cell)
    local18 = extract_local_geometry_features(current, c1x, c1y, cell, width, height)
    log_wh = size_trees.predict(np.concatenate([feat15, local18]).reshape(1, -1))[0]
    return cx, cy, math.exp(float(log_wh[0])) * cell, math.exp(float(log_wh[1])) * cell


def _emit_box(ws, we, cx, cy, w, h, conf) -> tuple:
    return (
        int(ws),
        we,
        int(round(cx)),
        int(round(cy)),
        int(round(w)),
        int(round(h)),
        conf,
    )


def _diag_update(window_gts, kept, counters: dict) -> None:
    if window_gts:
        counters["pos_windows"] += 1
        hit = False
        for gt in window_gts:
            for cx, cy, w, h, conf, _ in kept:
                ax1, ay1 = cx - w / 2, cy - h / 2
                ax2, ay2 = cx + w / 2, cy + h / 2
                bx1, by1 = gt.cx - gt.width / 2, gt.cy - gt.height / 2
                bx2, by2 = gt.cx + gt.width / 2, gt.cy + gt.height / 2
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = w * h + gt.width * gt.height - inter
                if union > 0 and inter / union >= 0.5:
                    hit = True
                    break
            if hit:
                break
        if hit:
            counters["pos_loc_hits"] += 1
    else:
        counters["empty_windows"] += 1
        if kept:
            counters["empty_fps"] += 1


def run_p_policies_shared(
    sequences: list[str],
    split_dir: Path,
    conf_model,
    size_trees,
    threshold: float,
):
    """One pass over windows; emit P1/P3/P5 with identical features/geometry math."""
    policies = ("P1", "P3", "P5")
    preds = {p: defaultdict(list) for p in policies}
    counters = {
        p: {"pos_windows": 0, "pos_loc_hits": 0, "empty_windows": 0, "empty_fps": 0}
        for p in policies
    }
    max_k = {"P1": 1, "P3": 3, "P5": 5}

    for sequence in sequences:
        print(f"    seq {sequence}", flush=True)
        stream = SequenceStream(sequence, split_dir)
        gts = read_detection_file(split_dir / f"{sequence}_bb_windows_40ms.txt")
        for ws in enumerate_challenge_windows(stream.timestamps):
            we = int(ws) + WINDOW_US
            current, prior = stream.slice_window(int(ws), we)
            candidates = stream.proposer.propose(current)
            window_gts = [g for g in gts if g.start_us < we and g.end_us > int(ws)]
            if not candidates:
                for p in policies:
                    _diag_update(window_gts, [], counters[p])
                continue
            features = extract_candidate_features_fast(
                current, prior, candidates, stream.width, stream.height, stream.cell
            )
            probs = conf_model.predict_proba(features)[:, 1]
            order = np.argsort(-probs)
            # Geometry for all candidates that could be emitted by P5 after NMS.
            shortlist = []
            for i in order:
                if float(probs[i]) < threshold:
                    break
                shortlist.append(int(i))
                if len(shortlist) >= 25:
                    break
            dets = []
            for i in shortlist:
                cx, cy, w, h = classical_box_from_candidate(
                    current,
                    candidates[i],
                    features[i],
                    stream.cell,
                    stream.width,
                    stream.height,
                    size_trees,
                )
                dets.append((cx, cy, w, h, float(probs[i]), candidates[i]))

            kept_by = {
                "P1": dets[:1],
                "P3": greedy_nms(dets, NMS_IOU)[: max_k["P3"]],
                "P5": greedy_nms(dets, NMS_IOU)[: max_k["P5"]],
            }
            for policy in policies:
                kept = kept_by[policy]
                for cx, cy, w, h, conf, _cand in kept:
                    preds[policy][sequence].append(_emit_box(ws, we, cx, cy, w, h, conf))
                _diag_update(window_gts, kept, counters[policy])

    out = {}
    for policy in policies:
        c = counters[policy]
        diag = {
            "positive_window_localization_recall": c["pos_loc_hits"] / max(c["pos_windows"], 1),
            "empty_window_false_positive_rate": c["empty_fps"] / max(c["empty_windows"], 1),
            "n_positive_windows": c["pos_windows"],
            "n_empty_windows": c["empty_windows"],
        }
        out[policy] = (dict(preds[policy]), diag)
    return out


LATENCY_SEQS = [
    "DAVIS_EGS_16908_2024-11-01-19-10-44",
    "DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06",
    "DVX_Filtered_Stars_2025-01-20-19-15-10",
    "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17",
    "2025_12_23_21_12_28_EVK4_mag5.2",
]


def benchmark_policy_latency(
    policy: str,
    sequences: list[str],
    split_dir: Path,
    conf_model,
    size_trees,
    threshold: float,
    max_windows: int = 500,
    ranker_bundle=None,
    onnx_session=None,
):
    """Complete-path latency for one policy (includes slicing through box emit)."""
    latency: dict[str, list[float]] = defaultdict(list)
    for sequence in sequences:
        stream = SequenceStream(sequence, split_dir)
        n_seq = 0
        for ws in enumerate_challenge_windows(stream.timestamps):
            if n_seq >= max_windows:
                break
            we = int(ws) + WINDOW_US
            t0 = perf_counter_ns()
            current, prior = stream.slice_window(int(ws), we)
            candidates = stream.proposer.propose(current)
            if not candidates:
                dt = (perf_counter_ns() - t0) / 1e6
                latency[stream.sensor].append(dt)
                latency["ALL"].append(dt)
                n_seq += 1
                continue
            features = extract_candidate_features_fast(
                current, prior, candidates, stream.width, stream.height, stream.cell
            )
            probs = conf_model.predict_proba(features)[:, 1]
            if policy == "H1_P1":
                ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
                scores = score_ranker(ranker_bundle, features, ranks)
                order = np.argsort(-scores)
                top3 = [candidates[int(i)] for i in order[: min(3, len(candidates))]]
                patches = [
                    rasterize_event_patch(
                        current, prior, c.cx, c.cy, float(stream.cell), int(ws), we
                    )
                    for c in top3
                ]
                feats3 = np.stack([features[candidates.index(c)] for c in top3]).astype(np.float32)
                cls_logits, _ = onnx_session.run(
                    None, {"patch": np.stack(patches).astype(np.float32), "features": feats3}
                )
                pick = int(np.argmax(cls_logits))
                sel = top3[pick]
                sel_i = candidates.index(sel)
                conf = float(probs[sel_i])
                if conf >= threshold:
                    classical_box_from_candidate(
                        current, sel, features[sel_i], stream.cell, stream.width, stream.height, size_trees
                    )
            else:
                max_k = {"P1": 1, "P3": 3, "P5": 5}[policy]
                order = np.argsort(-probs)
                shortlist = []
                for i in order:
                    if float(probs[i]) < threshold:
                        break
                    shortlist.append(int(i))
                    if len(shortlist) >= max(max_k * 5, max_k):
                        break
                dets = []
                for i in shortlist:
                    cx, cy, w, h = classical_box_from_candidate(
                        current,
                        candidates[i],
                        features[i],
                        stream.cell,
                        stream.width,
                        stream.height,
                        size_trees,
                    )
                    dets.append((cx, cy, w, h, float(probs[i]), candidates[i]))
                if policy == "P1":
                    _ = dets[:1]
                else:
                    _ = greedy_nms(dets, NMS_IOU)[:max_k]
            dt = (perf_counter_ns() - t0) / 1e6
            latency[stream.sensor].append(dt)
            latency["ALL"].append(dt)
            n_seq += 1
    return latency


def run_policy_on_sequences(
    sequences: list[str],
    split_dir: Path,
    conf_model,
    size_trees,
    threshold: float,
    policy: str,
    collect_latency: bool = False,
    ranker_bundle=None,
    onnx_session=None,
):
    """Return pred rows per sequence and optional latency samples.

    Policies P1/P3/P5: confidence ExtraTrees selects candidates; C4+S2 boxes.
    Policy H1_P1: ExtraTrees Top-3 → neural cls select → C4+S2; emit if conf>=thr.
    """
    preds: dict[str, list[tuple]] = defaultdict(list)
    latency: dict[str, list[float]] = defaultdict(list)
    counters = {"pos_windows": 0, "pos_loc_hits": 0, "empty_windows": 0, "empty_fps": 0}
    max_k = {"P1": 1, "P3": 3, "P5": 5, "H1_P1": 1}[policy]

    for sequence in sequences:
        stream = SequenceStream(sequence, split_dir)
        gt_path = split_dir / f"{sequence}_bb_windows_40ms.txt"
        gts = read_detection_file(gt_path)

        for ws in enumerate_challenge_windows(stream.timestamps):
            we = int(ws) + WINDOW_US
            t0 = perf_counter_ns()
            current, prior = stream.slice_window(int(ws), we)
            candidates = stream.proposer.propose(current)
            window_gts = [g for g in gts if g.start_us < we and g.end_us > int(ws)]
            if not candidates:
                if collect_latency:
                    dt = (perf_counter_ns() - t0) / 1e6
                    latency[stream.sensor].append(dt)
                    latency["ALL"].append(dt)
                _diag_update(window_gts, [], counters)
                continue

            features = extract_candidate_features_fast(
                current, prior, candidates, stream.width, stream.height, stream.cell
            )
            probs = conf_model.predict_proba(features)[:, 1]

            kept: list[tuple] = []
            if policy == "H1_P1":
                assert ranker_bundle is not None and onnx_session is not None
                ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
                scores = score_ranker(ranker_bundle, features, ranks)
                order = np.argsort(-scores)
                top3 = [candidates[int(i)] for i in order[: min(3, len(candidates))]]
                patches = [
                    rasterize_event_patch(
                        current, prior, c.cx, c.cy, float(stream.cell), int(ws), we
                    )
                    for c in top3
                ]
                feats3 = np.stack([features[candidates.index(c)] for c in top3]).astype(np.float32)
                cls_logits, _ = onnx_session.run(
                    None, {"patch": np.stack(patches).astype(np.float32), "features": feats3}
                )
                pick = int(np.argmax(cls_logits))
                sel = top3[pick]
                sel_i = candidates.index(sel)
                conf = float(probs[sel_i])
                if conf >= threshold:
                    cx, cy, w, h = classical_box_from_candidate(
                        current, sel, features[sel_i], stream.cell, stream.width, stream.height, size_trees
                    )
                    kept = [(cx, cy, w, h, conf, sel)]
            else:
                order = np.argsort(-probs)
                shortlist = []
                for i in order:
                    if float(probs[i]) < threshold:
                        break
                    shortlist.append(int(i))
                    if len(shortlist) >= max(max_k * 5, max_k):
                        break
                dets = []
                for i in shortlist:
                    cx, cy, w, h = classical_box_from_candidate(
                        current,
                        candidates[i],
                        features[i],
                        stream.cell,
                        stream.width,
                        stream.height,
                        size_trees,
                    )
                    dets.append((cx, cy, w, h, float(probs[i]), candidates[i]))
                if policy == "P1":
                    kept = dets[:1]
                else:
                    kept = greedy_nms(dets, NMS_IOU)[:max_k]

            for cx, cy, w, h, conf, _cand in kept:
                preds[sequence].append(_emit_box(ws, we, cx, cy, w, h, conf))

            if collect_latency:
                dt = (perf_counter_ns() - t0) / 1e6
                latency[stream.sensor].append(dt)
                latency["ALL"].append(dt)

            _diag_update(window_gts, kept, counters)

    diag = {
        "positive_window_localization_recall": counters["pos_loc_hits"] / max(counters["pos_windows"], 1),
        "empty_window_false_positive_rate": counters["empty_fps"] / max(counters["empty_windows"], 1),
        "n_positive_windows": counters["pos_windows"],
        "n_empty_windows": counters["empty_windows"],
    }
    return preds, latency, diag


def score_predictions(preds: dict[str, list[tuple]], sequences: list[str], split_dir: Path) -> dict:
    per_seq = {}
    for sequence in sequences:
        gt = load_tii_gt(split_dir / f"{sequence}_bb_windows_40ms.txt")
        # Cast pred centres/sizes as TII load_pred would (float then used as numbers)
        pred = preds.get(sequence, [])
        per_seq[sequence] = match_and_score(gt, pred)
    overall = aggregate_tii(per_seq)
    # sequence macro F1 / recall
    f1s = [m.f1 for m in per_seq.values() if m.n_gt > 0]
    recs = [m.recall for m in per_seq.values() if m.n_gt > 0]
    return {
        "overall": overall,
        "per_seq": per_seq,
        "sequence_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "sequence_macro_recall": float(np.mean(recs)) if recs else 0.0,
    }


def run_tii_official(gt_dir: Path, pred_dir: Path, excel_out: Path) -> dict[str, float]:
    """Run official TII evaluator; return overall Precision/Recall/F1/mAP from stdout."""
    import os

    cmd = [
        sys.executable,
        str(TII_EVAL),
        "--gt-dir",
        str(gt_dir),
        "--pred-dir",
        str(pred_dir),
        "--iou",
        "0.5",
        "--excel-out",
        str(excel_out.resolve()),
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        cmd,
        check=False,
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    safe = text.encode("ascii", errors="replace").decode("ascii")
    print(safe, flush=True)
    if proc.returncode != 0:
        # Excel may still exist if failure was only the post-save unicode print.
        if not excel_out.exists():
            raise RuntimeError(f"TII evaluate.py exit={proc.returncode}\n{safe}")
    out: dict[str, float] = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("| Precision"):
            parts = [p.strip() for p in s.split("|") if p.strip()]
            if len(parts) >= 2:
                out["precision"] = float(parts[1])
        elif s.startswith("| Recall"):
            parts = [p.strip() for p in s.split("|") if p.strip()]
            if len(parts) >= 2:
                out["recall"] = float(parts[1])
        elif s.startswith("| F1 Score"):
            parts = [p.strip() for p in s.split("|") if p.strip()]
            if len(parts) >= 2:
                out["f1"] = float(parts[1])
        elif "mAP @ IoU 0.5" in s:
            parts = [p.strip() for p in s.split("|") if p.strip()]
            if len(parts) >= 2:
                out["map50"] = float(parts[1])
    return out


def assert_tii_local_parity(local: TIIMetrics, tii: dict[str, float], fold_id: int, policy: str) -> None:
    """STOP if official TII and local evaluator disagree beyond numerical tolerance."""
    checks = [
        ("precision", local.precision, tii.get("precision")),
        ("recall", local.recall, tii.get("recall")),
        ("f1", local.f1, tii.get("f1")),
        ("ap50/map50", local.ap50, tii.get("map50")),
    ]
    for name, a, b in checks:
        if b is None:
            print(f"STOP parity: missing TII {name} fold={fold_id} policy={policy}", flush=True)
            raise SystemExit(3)
        if abs(float(a) - float(b)) > 1e-4:
            print(
                f"STOP parity discrepancy: fold={fold_id} policy={policy} {name} "
                f"local={a} tii={b}",
                flush=True,
            )
            raise SystemExit(3)


def pct(vals, p):
    return float(np.percentile(vals, p)) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=SPLIT)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/all_window_candidates.npz"))
    parser.add_argument("--table", type=Path, default=Path("artifacts/candidate_table.csv"))
    parser.add_argument("--folds", type=Path, default=Path("sequence_folds.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/runs/2026-08-30/challenge_metric_baseline"))
    parser.add_argument("--h1-gate", type=Path, default=Path("docs/runs/2026-08-30/challenge_metric_baseline/h1_gate.txt"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache)
    table = load_table(args.table)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))

    run_h1 = False
    if args.h1_gate.exists():
        run_h1 = "H1_beats_B_CURRENT=True" in args.h1_gate.read_text(encoding="utf-8")
    print(f"PART9 run_h1={run_h1}", flush=True)

    policy_summaries = []
    latency_rows = []
    diag_rows = []
    fold_rows = []
    seq_rows = []
    sensor_rows = []
    parity_ok = True

    for fold in folds:
        fold_id = int(fold["fold"])
        train_seqs = list(fold["train"])
        val_seqs = list(fold["validation"])
        print(f"Challenge fold {fold_id}: threshold OOF...", flush=True)
        thr = oof_threshold_on_train(cache, train_seqs)
        print(f"  threshold={thr:.6f}", flush=True)

        train_mask = np.array([str(s) in set(train_seqs) for s in cache["sequence"]], dtype=bool)
        conf_model = fit_confidence(cache["features"][train_mask], cache["target"][train_mask])

        train_idx = np.flatnonzero(np.isin(table["sequence"], train_seqs))
        size_trees = fit_size_s2(table, train_idx, args.split_dir)

        policies = ["P1", "P3", "P5"]
        ranker_bundle = None
        onnx_sess = None
        if run_h1:
            policies.append("H1_P1")
            rankers = fit_rankers(
                table["X"][train_idx],
                table["target"][train_idx],
                table["rank"][train_idx],
                model_names=[RANKER],
            )
            ranker_bundle = rankers[RANKER]
            import onnxruntime as ort

            onnx_path = Path("artifacts/tiny_foveated_onnx") / f"fold{fold_id}_refiner.onnx"
            onnx_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        print("  shared P1/P3/P5 inference...", flush=True)
        shared = run_p_policies_shared(val_seqs, args.split_dir, conf_model, size_trees, thr)
        policy_results = {p: shared[p] for p in ("P1", "P3", "P5")}
        if run_h1:
            print("  policy H1_P1...", flush=True)
            preds_h1, _, diag_h1 = run_policy_on_sequences(
                val_seqs,
                args.split_dir,
                conf_model,
                size_trees,
                thr,
                "H1_P1",
                collect_latency=False,
                ranker_bundle=ranker_bundle,
                onnx_session=onnx_sess,
            )
            policy_results["H1_P1"] = (preds_h1, diag_h1)

        # Latency on Part-1 representative sequences (complete path per policy).
        lat_seqs = [s for s in LATENCY_SEQS if (args.split_dir / f"{s}_labeled_events.npy").exists()]
        for policy in policies:
            print(f"  latency {policy}...", flush=True)
            latency = benchmark_policy_latency(
                policy,
                lat_seqs,
                args.split_dir,
                conf_model,
                size_trees,
                thr,
                max_windows=500,
                ranker_bundle=ranker_bundle,
                onnx_session=onnx_sess,
            )
            for sensor, vals in latency.items():
                latency_rows.append(
                    {
                        "fold": fold_id,
                        "policy": policy,
                        "sensor": sensor,
                        "p50_ms": pct(vals, 50),
                        "p95_ms": pct(vals, 95),
                        "p99_ms": pct(vals, 99),
                        "n": len(vals),
                    }
                )

        for policy in policies:
            print(f"  evaluate {policy}...", flush=True)
            preds, diag = policy_results[policy]
            scored = score_predictions(preds, val_seqs, args.split_dir)
            ov = scored["overall"]

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                gt_link = tmp_path / "gt"
                pred_dir = tmp_path / "pred"
                gt_link.mkdir()
                pred_dir.mkdir()
                for sequence in val_seqs:
                    src = args.split_dir / f"{sequence}_bb_windows_40ms.txt"
                    shutil.copy(src, gt_link / src.name)
                    write_tii_prediction_file(pred_dir / src.name, preds.get(sequence, []))
                local = evaluate_dirs_tii(gt_link, pred_dir)
                local_agg = aggregate_tii(local)
                if abs(local_agg.f1 - ov.f1) > 1e-6 or abs(local_agg.precision - ov.precision) > 1e-6:
                    parity_ok = False
                    print(
                        f"STOP parity local vs scored: fold={fold_id} policy={policy} "
                        f"f1 {local_agg.f1} vs {ov.f1}",
                        flush=True,
                    )
                    raise SystemExit(3)
                excel = args.out_dir / f"tii_fold{fold_id}_{policy}.xlsx"
                try:
                    tii_metrics = run_tii_official(gt_link, pred_dir, excel)
                except Exception as exc:
                    print(f"STOP TII evaluator failed: {exc}", flush=True)
                    raise SystemExit(3) from exc
                assert_tii_local_parity(local_agg, tii_metrics, fold_id, policy)

            fold_rows.append(
                {
                    "fold": fold_id,
                    "policy": policy,
                    "threshold": thr,
                    "precision": ov.precision,
                    "recall": ov.recall,
                    "f1": ov.f1,
                    "ap50": ov.ap50,
                    "tp": ov.tp,
                    "fp": ov.fp,
                    "fn": ov.fn,
                    "mean_matched_iou": ov.mean_matched_iou,
                    "sequence_macro_f1": scored["sequence_macro_f1"],
                    "sequence_macro_recall": scored["sequence_macro_recall"],
                    "pos_loc_recall": diag["positive_window_localization_recall"],
                    "empty_fp_rate": diag["empty_window_false_positive_rate"],
                }
            )
            for sequence, m in scored["per_seq"].items():
                seq_rows.append(
                    {
                        "fold": fold_id,
                        "policy": policy,
                        "sequence": sequence,
                        "sensor": sensor_name(sequence),
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "ap50": m.ap50,
                        "tp": m.tp,
                        "fp": m.fp,
                        "fn": m.fn,
                    }
                )
            diag_rows.append({"fold": fold_id, "policy": policy, **diag})

        write_csv(args.out_dir / "challenge_by_fold.csv", fold_rows)
        write_csv(args.out_dir / "challenge_by_sequence.csv", seq_rows)
        write_csv(args.out_dir / "challenge_latency.csv", latency_rows)
        write_csv(args.out_dir / "challenge_diagnostics.csv", diag_rows)
        print(f"  fold {fold_id} checkpointed", flush=True)

    # Aggregate policies across folds (micro by summing TP/FP/FN)
    all_policies = sorted({r["policy"] for r in fold_rows})
    for policy in all_policies:
        rows = [r for r in fold_rows if r["policy"] == policy]
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        ap = float(np.nanmean([r["ap50"] for r in rows]))
        policy_summaries.append(
            {
                "policy": policy,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "ap50_mean_folds": ap,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "sequence_macro_f1": float(np.mean([r["sequence_macro_f1"] for r in rows])),
                "sequence_macro_recall": float(np.mean([r["sequence_macro_recall"] for r in rows])),
                "pos_loc_recall": float(np.mean([r["pos_loc_recall"] for r in rows])),
                "empty_fp_rate": float(np.mean([r["empty_fp_rate"] for r in rows])),
            }
        )
    write_csv(args.out_dir / "challenge_policy_summary.csv", policy_summaries)

    # Per-sensor pooled from sequence rows
    for policy in all_policies:
        for sensor in ("DAVIS", "DVX", "EVK4"):
            rows = [r for r in seq_rows if r["policy"] == policy and r["sensor"] == sensor]
            if not rows:
                continue
            tp = sum(r["tp"] for r in rows)
            fp = sum(r["fp"] for r in rows)
            fn = sum(r["fn"] for r in rows)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            sensor_rows.append(
                {
                    "policy": policy,
                    "sensor": sensor,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "ap50_mean": float(np.nanmean([r["ap50"] for r in rows])),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
    write_csv(args.out_dir / "challenge_by_sensor.csv", sensor_rows)

    criteria = []
    for policy in all_policies:
        all_p95 = [r["p95_ms"] for r in latency_rows if r["policy"] == policy and r["sensor"] == "ALL"]
        overall_p95 = float(np.mean(all_p95)) if all_p95 else float("nan")
        lat_a = bool(overall_p95 <= 40.0)
        criteria.append({"policy": policy, "criterion": "LATENCY_A", "p95_ms": overall_p95, "pass": lat_a})
        sensor_ok = True
        for sensor in ("DAVIS", "DVX", "EVK4"):
            sp = [r["p95_ms"] for r in latency_rows if r["policy"] == policy and r["sensor"] == sensor]
            sp95 = float(np.mean(sp)) if sp else float("nan")
            ok = bool(sp95 <= 40.0) if not math.isnan(sp95) else False
            sensor_ok = sensor_ok and ok
            criteria.append({"policy": policy, "criterion": f"LATENCY_B_{sensor}", "p95_ms": sp95, "pass": ok})
        criteria.append({"policy": policy, "criterion": "LATENCY_B", "p95_ms": overall_p95, "pass": sensor_ok})
    write_csv(args.out_dir / "latency_criteria.csv", criteria)
    print("challenge_metric_done", flush=True)
    print(f"parity_ok={parity_ok}", flush=True)


if __name__ == "__main__":
    main()
