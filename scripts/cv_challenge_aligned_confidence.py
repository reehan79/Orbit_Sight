from __future__ import annotations

"""Challenge-aligned confidence experiment (Parts 3–10)."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from orbitsight.evaluation.tii_style import (
    aggregate_tii,
    evaluate_dirs_tii,
    load_tii_gt,
    match_and_score,
    pooled_percentiles,
    preds_from_thresholded_windows,
    select_detection_f1_threshold,
    select_f1_threshold,
    tii_iou,
    windows_overlap,
    write_tii_prediction_file,
)
from orbitsight.features import FEATURE_NAMES
from orbitsight.inference.b_current import SequenceStream
from orbitsight.inference.p1_detector import (
    benchmark_p1_latency,
    build_gate_features,
    emit_tii_row,
    run_p1_window_fast,
    run_p1_window_reference,
)
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows
from orbitsight.io import read_detection_file

PRIOR_MS = 80
TOP_K = 20
SPLIT = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
)
TII_EVAL = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_DataLoader\evaluate.py"
)
LATENCY_SEQS = [
    "DAVIS_EGS_16908_2024-11-01-19-10-44",
    "DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06",
    "DVX_Filtered_Stars_2025-01-20-19-15-10",
    "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17",
    "2025_12_23_21_12_28_EVK4_mag5.2",
]


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


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_latency_fold(out_dir: Path, fold_id: int, latency_samples: dict) -> None:
    payload = {
        method: {sensor: vals for sensor, vals in sensors.items()}
        for method, sensors in latency_samples.items()
    }
    path = out_dir / f"latency_raw_fold{fold_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_latency_fold(out_dir: Path, fold_id: int) -> dict[str, dict[str, list[float]]]:
    path = out_dir / f"latency_raw_fold{fold_id}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        method: {sensor: [float(v) for v in vals] for sensor, vals in sensors.items()}
        for method, sensors in payload.items()
    }


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


def fit_size_s2(table, train_idx, split_dir: Path):
    from orbitsight.features import extract_local_geometry_features, refine_c1_centroid
    from orbitsight.proposals import RawGridProposer, infer_sensor_geometry

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


def oof_candidate_f1_threshold(cache: dict, train_sequences: list[str]) -> float:
    mask = np.array([str(s) in set(train_sequences) for s in cache["sequence"]], dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0.5
    seqs = np.unique(cache["sequence"][idx].astype(str))
    if len(seqs) < 2:
        clf = fit_confidence(cache["features"][idx], cache["target"][idx])
        scores = clf.predict_proba(cache["features"][idx])[:, 1]
        return select_f1_threshold(cache["target"][idx], scores)
    kf = KFold(n_splits=min(5, len(seqs)), shuffle=True, random_state=42)
    oof_scores = np.zeros(len(idx), dtype=np.float64)
    oof_y = cache["target"][idx].astype(np.int8)
    seq_of = cache["sequence"][idx].astype(str)
    for train_s, val_s in kf.split(seqs):
        train_set, val_set = set(seqs[train_s]), set(seqs[val_s])
        tr = np.array([s in train_set for s in seq_of])
        va = np.array([s in val_set for s in seq_of])
        if not tr.any() or not va.any():
            continue
        clf = fit_confidence(cache["features"][idx[tr]], cache["target"][idx[tr]])
        oof_scores[va] = clf.predict_proba(cache["features"][idx[va]])[:, 1]
    return select_f1_threshold(oof_y, oof_scores)


@dataclass
class WindowRecord:
    sequence: str
    ws: int
    we: int
    row: tuple
    confidence: float
    has_gt: bool
    is_tp_if_emitted: bool
    gate_features: np.ndarray | None = None


def window_gts_from_file(gts, ws: int, we: int):
    return [g for g in gts if g.start_us < we and g.end_us > ws]


def is_tp_box(ws, we, cx, cy, w, h, window_gts) -> bool:
    pred_box = (cx, cy, w, h)
    for g in window_gts:
        gt_box = (g.cx, g.cy, g.width, g.height)
        if windows_overlap(ws, we, g.start_us, g.end_us) and tii_iou(pred_box, gt_box) >= 0.5:
            return True
    return False


def run_unthresholded_p1(
    sequences: list[str],
    split_dir: Path,
    conf_model,
    size_trees,
    build_gates: bool = False,
) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    for sequence in sequences:
        stream = SequenceStream(sequence, split_dir)
        det_gts = read_detection_file(split_dir / f"{sequence}_bb_windows_40ms.txt")
        for ws in enumerate_challenge_windows(stream.timestamps):
            we = int(ws) + WINDOW_US
            res = run_p1_window_reference(
                stream, int(ws), we, conf_model, size_trees, always_emit=True
            )
            if res is None:
                continue
            row = emit_tii_row(res)
            wgts = window_gts_from_file(det_gts, int(ws), we)
            has_gt = len(wgts) > 0
            is_tp = is_tp_box(int(ws), we, row[2], row[3], row[4], row[5], wgts) if has_gt else False
            gf = build_gate_features(res, stream, size_trees) if build_gates else None
            records.append(
                WindowRecord(
                    sequence=sequence,
                    ws=int(ws),
                    we=we,
                    row=row,
                    confidence=float(res.confidence),
                    has_gt=has_gt,
                    is_tp_if_emitted=is_tp,
                    gate_features=gf,
                )
            )
    return records


def records_to_preds(records: list[WindowRecord], threshold: float | None = None, scores: np.ndarray | None = None):
    preds: dict[str, list[tuple]] = defaultdict(list)
    for i, rec in enumerate(records):
        score = float(scores[i]) if scores is not None else rec.confidence
        if threshold is None or score >= threshold:
            ws, we, cx, cy, w, h, _ = rec.row
            preds[rec.sequence].append((ws, we, cx, cy, w, h, score))
    return dict(preds)


def ap_preds_from_records(records: list[WindowRecord], scores: np.ndarray | None = None):
    """AP uses continuous scores; all top-1 boxes retained."""
    preds: dict[str, list[tuple]] = defaultdict(list)
    for i, rec in enumerate(records):
        score = float(scores[i]) if scores is not None else rec.confidence
        ws, we, cx, cy, w, h, _ = rec.row
        preds[rec.sequence].append((ws, we, cx, cy, w, h, score))
    return dict(preds)


def score_preds(preds: dict[str, list[tuple]], sequences: list[str], split_dir: Path) -> dict:
    per_seq = {}
    for sequence in sequences:
        gt = load_tii_gt(split_dir / f"{sequence}_bb_windows_40ms.txt")
        pred = preds.get(sequence, [])
        per_seq[sequence] = match_and_score(gt, pred)
    overall = aggregate_tii(per_seq)
    f1s = [m.f1 for m in per_seq.values() if m.n_gt > 0]
    recs = [m.recall for m in per_seq.values() if m.n_gt > 0]
    return {
        "overall": overall,
        "per_seq": per_seq,
        "sequence_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "sequence_macro_recall": float(np.mean(recs)) if recs else 0.0,
    }


def diag_from_records(records: list[WindowRecord], threshold: float, scores: np.ndarray | None = None):
    pos_w, pos_hit, empty_w, empty_fp = 0, 0, 0, 0
    for i, rec in enumerate(records):
        score = float(scores[i]) if scores is not None else rec.confidence
        emit = score >= threshold
        if rec.has_gt:
            pos_w += 1
            if emit and rec.is_tp_if_emitted:
                pos_hit += 1
        else:
            empty_w += 1
            if emit:
                empty_fp += 1
    return {
        "positive_window_localization_recall": pos_hit / max(pos_w, 1),
        "empty_window_false_positive_rate": empty_fp / max(empty_w, 1),
    }


def inner_oof_windows(
    outer_train: list[str],
    cache: dict,
    table: dict,
    split_dir: Path,
    build_gates: bool,
) -> tuple[list[WindowRecord], list[WindowRecord]]:
    """Inner-OOF val records + inner-train gate rows from outer train only."""
    seqs = np.array(sorted(set(outer_train)))
    if len(seqs) < 2:
        train_idx = np.flatnonzero(np.isin(table["sequence"], outer_train))
        mask = np.array([str(s) in set(outer_train) for s in cache["sequence"]], dtype=bool)
        conf = fit_confidence(cache["features"][mask], cache["target"][mask])
        size = fit_size_s2(table, train_idx, split_dir)
        val_recs = run_unthresholded_p1(list(outer_train), split_dir, conf, size, build_gates)
        return val_recs, []
    kf = KFold(n_splits=min(5, len(seqs)), shuffle=True, random_state=42)
    val_records: list[WindowRecord] = []
    train_gate_records: list[WindowRecord] = []
    for tr_i, va_i in kf.split(seqs):
        inner_train = list(seqs[tr_i])
        inner_val = list(seqs[va_i])
        train_idx = np.flatnonzero(np.isin(table["sequence"], inner_train))
        mask = np.array([str(s) in set(inner_train) for s in cache["sequence"]], dtype=bool)
        conf = fit_confidence(cache["features"][mask], cache["target"][mask])
        size = fit_size_s2(table, train_idx, split_dir)
        val_records.extend(run_unthresholded_p1(inner_val, split_dir, conf, size, build_gates))
        if build_gates:
            train_gate_records.extend(run_unthresholded_p1(inner_train, split_dir, conf, size, True))
    return val_records, train_gate_records


def detection_f1_threshold_from_records(records: list[WindowRecord], scores: np.ndarray) -> float:
    emit_mask = np.ones(len(records), dtype=bool)
    has_gt = np.array([r.has_gt for r in records], dtype=bool)
    is_tp = np.array([r.is_tp_if_emitted for r in records], dtype=bool)
    return select_detection_f1_threshold(scores, emit_mask, is_tp, has_gt)


def fit_gate_g1(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(Xs, y)
    return scaler, clf


def fit_gate_g2(X: np.ndarray, y: np.ndarray):
    clf = ExtraTreesClassifier(
        n_estimators=64,
        max_depth=12,
        min_samples_leaf=20,
        max_features=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X, y)
    return clf


def score_gate_g1(scaler, clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(scaler.transform(X))[:, 1]


def score_gate_g2(clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)[:, 1]


def inner_oof_combined(
    outer_train: list[str],
    cache: dict,
    table: dict,
    split_dir: Path,
) -> tuple[list[WindowRecord], np.ndarray, np.ndarray, list[WindowRecord], list[WindowRecord]]:
    """Single inner-CV pass: val records, G1/G2 OOF scores, train gate rows."""
    seqs = np.array(sorted(set(outer_train)))
    if len(seqs) < 2:
        train_idx = np.flatnonzero(np.isin(table["sequence"], outer_train))
        mask = np.array([str(s) in set(outer_train) for s in cache["sequence"]], dtype=bool)
        conf = fit_confidence(cache["features"][mask], cache["target"][mask])
        size = fit_size_s2(table, train_idx, split_dir)
        val_recs = run_unthresholded_p1(list(outer_train), split_dir, conf, size, True)
        return val_recs, np.array([]), np.array([]), val_recs, val_recs

    kf = KFold(n_splits=min(5, len(seqs)), shuffle=True, random_state=42)
    val_records: list[WindowRecord] = []
    train_gate_g1: list[WindowRecord] = []
    train_gate_g2: list[WindowRecord] = []
    oof_g1: list[float] = []
    oof_g2: list[float] = []

    for tr_i, va_i in kf.split(seqs):
        inner_train = list(seqs[tr_i])
        inner_val = list(seqs[va_i])
        train_idx = np.flatnonzero(np.isin(table["sequence"], inner_train))
        mask = np.array([str(s) in set(inner_train) for s in cache["sequence"]], dtype=bool)
        conf = fit_confidence(cache["features"][mask], cache["target"][mask])
        size = fit_size_s2(table, train_idx, split_dir)
        tr_recs = run_unthresholded_p1(inner_train, split_dir, conf, size, True)
        va_recs = run_unthresholded_p1(inner_val, split_dir, conf, size, True)
        val_records.extend(va_recs)
        train_gate_g1.extend(tr_recs)
        train_gate_g2.extend(tr_recs)

        Xtr = np.stack([r.gate_features for r in tr_recs])
        ytr = np.array([r.is_tp_if_emitted for r in tr_recs], dtype=np.int8)
        g1_scaler, g1_clf = fit_gate_g1(Xtr, ytr)
        g2_clf = fit_gate_g2(Xtr, ytr)
        oof_g1.extend(score_gate_g1(g1_scaler, g1_clf, np.stack([r.gate_features for r in va_recs])).tolist())
        oof_g2.extend(score_gate_g2(g2_clf, np.stack([r.gate_features for r in va_recs])).tolist())

    return (
        val_records,
        np.asarray(oof_g1, dtype=np.float64),
        np.asarray(oof_g2, dtype=np.float64),
        train_gate_g1,
        train_gate_g2,
    )


def inner_oof_gate_scores(
    outer_train: list[str],
    cache: dict,
    table: dict,
    split_dir: Path,
    gate_kind: str,
) -> tuple[np.ndarray, list[WindowRecord], list[WindowRecord]]:
    seqs = np.array(sorted(set(outer_train)))
    if len(seqs) < 2:
        train_idx = np.flatnonzero(np.isin(table["sequence"], outer_train))
        mask = np.array([str(s) in set(outer_train) for s in cache["sequence"]], dtype=bool)
        conf = fit_confidence(cache["features"][mask], cache["target"][mask])
        size = fit_size_s2(table, train_idx, split_dir)
        val_recs = run_unthresholded_p1(list(outer_train), split_dir, conf, size, True)
        return np.array([]), val_recs, val_recs
    kf = KFold(n_splits=min(5, len(seqs)), shuffle=True, random_state=42)
    val_records: list[WindowRecord] = []
    train_gate_records: list[WindowRecord] = []
    oof_scores: list[float] = []
    for tr_i, va_i in kf.split(seqs):
        inner_train = list(seqs[tr_i])
        inner_val = list(seqs[va_i])
        train_idx = np.flatnonzero(np.isin(table["sequence"], inner_train))
        mask = np.array([str(s) in set(inner_train) for s in cache["sequence"]], dtype=bool)
        conf = fit_confidence(cache["features"][mask], cache["target"][mask])
        size = fit_size_s2(table, train_idx, split_dir)
        tr_recs = run_unthresholded_p1(inner_train, split_dir, conf, size, True)
        va_recs = run_unthresholded_p1(inner_val, split_dir, conf, size, True)
        Xtr = np.stack([r.gate_features for r in tr_recs])
        ytr = np.array([r.is_tp_if_emitted for r in tr_recs], dtype=np.int8)
        if gate_kind == "G1":
            scaler, clf = fit_gate_g1(Xtr, ytr)
            scores = score_gate_g1(scaler, clf, np.stack([r.gate_features for r in va_recs]))
        else:
            clf = fit_gate_g2(Xtr, ytr)
            scores = score_gate_g2(clf, np.stack([r.gate_features for r in va_recs]))
        val_records.extend(va_recs)
        train_gate_records.extend(tr_recs)
        oof_scores.extend(scores.tolist())
    return np.asarray(oof_scores, dtype=np.float64), val_records, train_gate_records


def run_tii_official(gt_dir: Path, pred_dir: Path, excel_out: Path) -> dict[str, float]:
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
    if proc.returncode != 0 and not excel_out.exists():
        raise RuntimeError(f"TII exit={proc.returncode}\n{safe}")
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


def assert_tii_parity(local, tii: dict[str, float], fold_id: int, label: str) -> None:
    for name, a, b in [
        ("precision", local.precision, tii.get("precision")),
        ("recall", local.recall, tii.get("recall")),
        ("f1", local.f1, tii.get("f1")),
        ("ap50", local.ap50, tii.get("map50")),
    ]:
        if b is None:
            raise SystemExit(f"STOP missing TII {name} fold={fold_id} {label}")
        if abs(float(a) - float(b)) > 1e-4:
            raise SystemExit(f"STOP parity {label} {name} local={a} tii={b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=SPLIT)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/all_window_candidates.npz"))
    parser.add_argument("--table", type=Path, default=Path("artifacts/candidate_table.csv"))
    parser.add_argument("--folds", type=Path, default=Path("sequence_folds.json"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/runs/2026-08-31/challenge_aligned_confidence"),
    )
    parser.add_argument(
        "--start-fold",
        type=int,
        default=-1,
        help="Skip folds < start-fold. Default: auto-resume from checkpoint CSVs.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache)
    table = load_table(args.table)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))

    # Resume from per-fold checkpoints if present.
    compare_fold_rows = read_csv_rows(args.out_dir / "compare_by_fold.csv")
    compare_seq_rows = read_csv_rows(args.out_dir / "compare_by_sequence.csv")
    threshold_rows = read_csv_rows(args.out_dir / "threshold_stability.csv")
    ceiling_rows = read_csv_rows(args.out_dir / "ceiling_summary.csv")
    gate_stats_rows = read_csv_rows(args.out_dir / "oof_gate_stats.csv")
    # Cast numeric fields that CSV read as strings for later aggregation.
    for rows in (compare_fold_rows, compare_seq_rows, ceiling_rows):
        for r in rows:
            for k, v in list(r.items()):
                if k in ("fold", "tp", "fp", "fn", "gate_train_rows", "gate_train_positives", "inner_oof_val_rows"):
                    try:
                        r[k] = int(float(v))
                    except (TypeError, ValueError):
                        pass
                elif k not in ("method", "mode", "sequence", "sensor"):
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
    for r in threshold_rows:
        r["fold"] = int(float(r["fold"]))
        r["threshold"] = float(r["threshold"])
    for r in gate_stats_rows:
        r["fold"] = int(float(r["fold"]))
        for k in ("gate_train_rows", "gate_train_positives", "inner_oof_val_rows"):
            r[k] = int(float(r[k]))

    done_folds = {int(r["fold"]) for r in compare_fold_rows}
    if args.start_fold >= 0:
        start_fold = args.start_fold
    else:
        start_fold = (max(done_folds) + 1) if done_folds else 0
    print(f"Resume: done_folds={sorted(done_folds)} start_fold={start_fold}", flush=True)

    latency_samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fid in sorted(done_folds):
        loaded = load_latency_fold(args.out_dir, fid)
        for method, sensors in loaded.items():
            for sensor, vals in sensors.items():
                latency_samples[method][sensor].extend(vals)

    for fold in folds:
        fold_id = int(fold["fold"])
        if fold_id < start_fold or fold_id in done_folds:
            print(f"Fold {fold_id}: skip (already complete)", flush=True)
            continue
        train_seqs = list(fold["train"])
        val_seqs = list(fold["validation"])
        print(f"Fold {fold_id} inner OOF...", flush=True)

        t0 = oof_candidate_f1_threshold(cache, train_seqs)
        inner_val_recs, oof_g1, oof_g2, train_gate_g1, train_gate_g2 = inner_oof_combined(
            train_seqs, cache, table, args.split_dir
        )
        conf_scores = np.array([r.confidence for r in inner_val_recs], dtype=np.float64)
        t1 = detection_f1_threshold_from_records(inner_val_recs, conf_scores)
        thr_g1 = detection_f1_threshold_from_records(inner_val_recs, oof_g1) if len(oof_g1) else 0.5
        thr_g2 = detection_f1_threshold_from_records(inner_val_recs, oof_g2) if len(oof_g2) else 0.5

        threshold_rows.extend(
            [
                {"fold": fold_id, "method": "D0_T0", "threshold": t0},
                {"fold": fold_id, "method": "D1_T1", "threshold": t1},
                {"fold": fold_id, "method": "D2_G1", "threshold": thr_g1},
                {"fold": fold_id, "method": "D3_G2", "threshold": thr_g2},
            ]
        )

        gate_stats_rows.append(
            {
                "fold": fold_id,
                "gate_train_rows": len(train_gate_g1),
                "gate_train_positives": int(sum(r.is_tp_if_emitted for r in train_gate_g1)),
                "inner_oof_val_rows": len(inner_val_recs),
            }
        )

        train_idx = np.flatnonzero(np.isin(table["sequence"], train_seqs))
        mask = np.array([str(s) in set(train_seqs) for s in cache["sequence"]], dtype=bool)
        conf_model = fit_confidence(cache["features"][mask], cache["target"][mask])
        size_trees = fit_size_s2(table, train_idx, args.split_dir)

        val_records = run_unthresholded_p1(val_seqs, args.split_dir, conf_model, size_trees, build_gates=True)
        ap_preds = ap_preds_from_records(val_records)

        train_gate_recs = train_gate_g1
        Xg = np.stack([r.gate_features for r in train_gate_recs])
        yg = np.array([r.is_tp_if_emitted for r in train_gate_recs], dtype=np.int8)
        g1_scaler, g1_clf = fit_gate_g1(Xg, yg)
        g2_clf = fit_gate_g2(
            np.stack([r.gate_features for r in train_gate_g2]),
            np.array([r.is_tp_if_emitted for r in train_gate_g2], dtype=np.int8),
        )
        gate_scores = score_gate_g1(g1_scaler, g1_clf, np.stack([r.gate_features for r in val_records]))
        gate_scores2 = score_gate_g2(g2_clf, np.stack([r.gate_features for r in val_records]))

        methods = {
            "D0": (None, t0),
            "D1": (None, t1),
            "D2": (gate_scores, thr_g1),
            "D3": (gate_scores2, thr_g2),
        }

        # PART 3 ceiling on validation
        always_preds = ap_preds_from_records(val_records)
        oracle_preds: dict[str, list[tuple]] = defaultdict(list)
        for rec in val_records:
            if rec.is_tp_if_emitted:
                ws, we, cx, cy, w, h, c = rec.row
                oracle_preds[rec.sequence].append((ws, we, cx, cy, w, h, rec.confidence))
        always_scored = score_preds(always_preds, val_seqs, args.split_dir)
        oracle_scored = score_preds(dict(oracle_preds), val_seqs, args.split_dir)
        for label, scored in [("P1_ALWAYS", always_scored), ("ORACLE_EMIT", oracle_scored)]:
            ov = scored["overall"]
            ceiling_rows.append(
                {
                    "fold": fold_id,
                    "mode": label,
                    "precision": ov.precision,
                    "recall": ov.recall,
                    "f1": ov.f1,
                    "tp": ov.tp,
                    "fp": ov.fp,
                    "fn": ov.fn,
                }
            )

        for method, (scores, thr) in methods.items():
            thr_preds = records_to_preds(val_records, thr, scores)
            scored_thr = score_preds(thr_preds, val_seqs, args.split_dir)
            scored_ap = score_preds(ap_preds, val_seqs, args.split_dir)
            ov = scored_thr["overall"]
            ap_ov = scored_ap["overall"]
            diag = diag_from_records(val_records, thr, scores)
            compare_fold_rows.append(
                {
                    "fold": fold_id,
                    "method": method,
                    "threshold": thr,
                    "precision": ov.precision,
                    "recall": ov.recall,
                    "f1": ov.f1,
                    "ap50": ap_ov.ap50,
                    "tp": ov.tp,
                    "fp": ov.fp,
                    "fn": ov.fn,
                    "mean_matched_iou": ov.mean_matched_iou,
                    "sequence_macro_f1": scored_thr["sequence_macro_f1"],
                    "sequence_macro_recall": scored_thr["sequence_macro_recall"],
                    "pos_loc_recall": diag["positive_window_localization_recall"],
                    "empty_fp_rate": diag["empty_window_false_positive_rate"],
                }
            )
            for sequence, m in scored_thr["per_seq"].items():
                compare_seq_rows.append(
                    {
                        "fold": fold_id,
                        "method": method,
                        "sequence": sequence,
                        "sensor": sensor_name(sequence),
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "ap50": scored_ap["per_seq"][sequence].ap50,
                        "tp": m.tp,
                        "fp": m.fp,
                        "fn": m.fn,
                    }
                )

            with tempfile.TemporaryDirectory() as tmp:
                gt_dir = Path(tmp) / "gt"
                pred_dir = Path(tmp) / "pred"
                gt_dir.mkdir()
                pred_dir.mkdir()
                for sequence in val_seqs:
                    shutil.copy(
                        args.split_dir / f"{sequence}_bb_windows_40ms.txt",
                        gt_dir / f"{sequence}_bb_windows_40ms.txt",
                    )
                    write_tii_prediction_file(
                        pred_dir / f"{sequence}_bb_windows_40ms.txt",
                        thr_preds.get(sequence, []),
                    )
                local = aggregate_tii(evaluate_dirs_tii(gt_dir, pred_dir))
                excel = args.out_dir / f"tii_fold{fold_id}_{method}.xlsx"
                tii = run_tii_official(gt_dir, pred_dir, excel)
                assert_tii_parity(local, tii, fold_id, method)

        # Latency PART 9
        lat_seqs = [s for s in LATENCY_SEQS if (args.split_dir / f"{s}_labeled_events.npy").exists()]
        fold_lat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for method, (scores, thr) in {"D1": (None, t1), "D2": (gate_scores, thr_g1), "D3": (gate_scores2, thr_g2)}.items():
            for seq in lat_seqs:
                stream = SequenceStream(seq, args.split_dir)
                if method == "D1":
                    samples = benchmark_p1_latency(stream, conf_model, size_trees, thr)
                elif method == "D2":
                    samples = benchmark_p1_latency(
                        stream, conf_model, size_trees, thr,
                        gate_scaler=g1_scaler, gate_clf=g1_clf, gate_threshold=thr,
                    )
                else:
                    samples = benchmark_p1_latency(
                        stream, conf_model, size_trees, thr,
                        gate_et=g2_clf, gate_threshold=thr,
                    )
                fold_lat[method][stream.sensor].extend(samples)
                fold_lat[method]["ALL"].extend(samples)
                latency_samples[method][stream.sensor].extend(samples)
                latency_samples[method]["ALL"].extend(samples)
        save_latency_fold(args.out_dir, fold_id, fold_lat)

        print(f"  fold {fold_id} done", flush=True)
        write_csv(args.out_dir / "compare_by_fold.csv", compare_fold_rows)
        write_csv(args.out_dir / "compare_by_sequence.csv", compare_seq_rows)
        write_csv(args.out_dir / "threshold_stability.csv", threshold_rows)
        write_csv(args.out_dir / "ceiling_summary.csv", ceiling_rows)
        write_csv(args.out_dir / "oof_gate_stats.csv", gate_stats_rows)

    write_csv(args.out_dir / "ceiling_summary.csv", ceiling_rows)
    write_csv(args.out_dir / "compare_by_fold.csv", compare_fold_rows)
    write_csv(args.out_dir / "compare_by_sequence.csv", compare_seq_rows)
    write_csv(args.out_dir / "threshold_stability.csv", threshold_rows)
    write_csv(args.out_dir / "oof_gate_stats.csv", gate_stats_rows)

    # Threshold stats
    stats_rows = []
    for method in ("D0_T0", "D1_T1", "D2_G1", "D3_G2"):
        vals = [r["threshold"] for r in threshold_rows if r["method"] == method]
        if vals:
            stats_rows.append(
                {
                    "method": method,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )
    write_csv(args.out_dir / "threshold_stats.csv", stats_rows)

    # Pooled latency + criteria
    latency_rows = []
    criteria_rows = []
    for method in ("D1", "D2", "D3"):
        pooled = pooled_percentiles(dict(latency_samples[method]))
        for sensor in ("ALL", "DAVIS", "DVX", "EVK4"):
            if sensor in latency_samples[method]:
                sp = pooled_percentiles({sensor: latency_samples[method][sensor]})
                latency_rows.append(
                    {
                        "method": method,
                        "sensor": sensor,
                        "p50_ms": sp["p50_ms"],
                        "p95_ms": sp["p95_ms"],
                        "p99_ms": sp["p99_ms"],
                        "n": sp["n"],
                    }
                )
        lat_a = pooled["p95_ms"] <= 40.0
        criteria_rows.append({"criterion": f"LATENCY_A_{method}", "value": pooled["p95_ms"], "pass": lat_a})
        sensor_ok = True
        for sensor in ("DAVIS", "DVX", "EVK4"):
            sp = pooled_percentiles({sensor: latency_samples[method][sensor]})
            ok = sp["p95_ms"] <= 40.0
            sensor_ok = sensor_ok and ok
            criteria_rows.append(
                {"criterion": f"LATENCY_B_{method}_{sensor}", "value": sp["p95_ms"], "pass": ok}
            )
        criteria_rows.append({"criterion": f"LATENCY_B_{method}", "value": pooled["p95_ms"], "pass": sensor_ok})

    write_csv(args.out_dir / "latency_pooled.csv", latency_rows)

    # Decision criteria CONF_A-D vs D0
    d0_f1 = [r["f1"] for r in compare_fold_rows if r["method"] == "D0"]
    d0_ap = [r["ap50"] for r in compare_fold_rows if r["method"] == "D0"]
    d0_fpr = [r["empty_fp_rate"] for r in compare_fold_rows if r["method"] == "D0"]
    d0_prec = [r["precision"] for r in compare_fold_rows if r["method"] == "D0"]
    d0_mean_f1 = float(np.mean(d0_f1))
    d0_mean_ap = float(np.nanmean(d0_ap))
    d0_mean_fpr = float(np.mean(d0_fpr))
    d0_mean_prec = float(np.mean(d0_prec))
    for method in ("D1", "D2", "D3"):
        rows = [r for r in compare_fold_rows if r["method"] == method]
        mf1 = float(np.mean([r["f1"] for r in rows]))
        map50 = float(np.nanmean([r["ap50"] for r in rows]))
        mfpr = float(np.mean([r["empty_fp_rate"] for r in rows]))
        mprec = float(np.mean([r["precision"] for r in rows]))
        criteria_rows.extend(
            [
                {"criterion": f"CONF_A_{method}", "value": mf1 - d0_mean_f1, "pass": mf1 >= d0_mean_f1 + 0.05},
                {"criterion": f"CONF_B_{method}", "value": map50 - d0_mean_ap, "pass": map50 >= d0_mean_ap + 0.03},
                {"criterion": f"CONF_C_{method}", "value": mfpr, "pass": mfpr <= 0.05},
                {"criterion": f"CONF_D_{method}", "value": mprec, "pass": mprec >= 0.40},
            ]
        )
    write_csv(args.out_dir / "decision_criteria.csv", criteria_rows)

    # Pooled compare summary
    summary = []
    for method in ("D0", "D1", "D2", "D3"):
        rows = [r for r in compare_fold_rows if r["method"] == method]
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        summary.append(
            {
                "method": method,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "ap50_mean_folds": float(np.nanmean([r["ap50"] for r in rows])),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "sequence_macro_f1": float(np.mean([r["sequence_macro_f1"] for r in rows])),
                "empty_fp_rate": float(np.mean([r["empty_fp_rate"] for r in rows])),
            }
        )
    write_csv(args.out_dir / "compare_policy_summary.csv", summary)

    sensor_rows = []
    for method in ("D0", "D1", "D2", "D3"):
        for sensor in ("DAVIS", "DVX", "EVK4"):
            rows = [r for r in compare_seq_rows if r["method"] == method and r["sensor"] == sensor]
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
                    "method": method,
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
    write_csv(args.out_dir / "compare_by_sensor.csv", sensor_rows)
    print("challenge_aligned_done", flush=True)


if __name__ == "__main__":
    main()
