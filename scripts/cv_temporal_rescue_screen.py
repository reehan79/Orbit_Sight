"""SCREEN: selective temporal rescue on D2-rejected windows (outer folds 1,4).

Does not modify the accepted D2 path. Rescue emits at most one box on reject.
Never access Testing_sets. Do not run full five-fold CV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Reuse champion helpers from the challenge-aligned CV script.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import cv_challenge_aligned_confidence as cv  # noqa: E402

from orbitsight.evaluation.tii_style import (
    aggregate_tii,
    evaluate_dirs_tii,
    match_and_score,
    load_tii_gt,
    pooled_percentiles,
    write_tii_prediction_file,
)
from orbitsight.features import FEATURE_NAMES
from orbitsight.inference.b_current import SequenceStream
from orbitsight.inference.p1_detector import (
    build_gate_features,
    emit_tii_row,
    run_p1_window_fast,
    run_p1_window_reference,
)
from orbitsight.inference.temporal_rescue import (
    HISTORY,
    CausalTemporalState,
    HistorySlot,
    N_TEMPORAL_FEATURES,
    TEMPORAL_FEATURE_NAMES,
)
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows
from orbitsight.io import read_detection_file
from orbitsight.sprint import parse_fold_ids, write_atomic_json, write_atomic_text

LOG_GLOBAL_IDX = FEATURE_NAMES.index("log_global_event_count")
SPLIT = cv.SPLIT
TII_EVAL = cv.TII_EVAL


def write_csv_union(path: Path, rows: list[dict]) -> None:
    """Write rows with union of keys (missing extras become empty)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


LATENCY_SEQS = [
    "DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06",
    "DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17",
    "DVX_Filtered_Stars_2025-01-20-19-15-10",
    "2025_12_23_21_12_28_EVK4_mag5.2",
]

SCREEN_FOLDS = [1, 4]


@dataclass
class ScoredWindow:
    sequence: str
    ws: int
    we: int
    row: tuple  # ws,we,cx,cy,w,h,base_conf (integer boxes)
    base_conf: float
    gate_score: float
    has_gt: bool
    is_tp_if_emitted: bool
    temporal: np.ndarray
    cx_cells: float
    cy_cells: float
    event_count: float
    top20_has_tp_geometry: bool | None = None


def fit_rescue(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(Xs, y)
    return scaler, clf


def score_rescue(scaler, clf, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.asarray([], dtype=np.float64)
    return clf.predict_proba(scaler.transform(X))[:, 1]


def _gt_near_any_candidate(res, wgts, cell: float) -> bool:
    """Diagnostic: any Top-20 candidate centre within ~2 cells of a GT centre."""
    if not wgts or not res.candidates:
        return False
    for g in wgts:
        gcx = float(g.cx) / cell
        gcy = float(g.cy) / cell
        for cand in res.candidates:
            if math.hypot(float(cand.cx) / cell - gcx, float(cand.cy) / cell - gcy) <= 2.0:
                return True
    return False


def run_scored_sequence(
    sequence: str,
    split_dir: Path,
    conf_model,
    size_trees,
    gate_scaler,
    gate_clf,
    *,
    use_fast: bool = False,
) -> list[ScoredWindow]:
    """Unthresholded TRUE-P1 + D2 gate score + causal temporal features."""
    stream = SequenceStream(sequence, split_dir)
    det_gts = read_detection_file(split_dir / f"{sequence}_bb_windows_40ms.txt")
    state = CausalTemporalState(HISTORY)
    run_fn = run_p1_window_fast if use_fast else run_p1_window_reference
    out: list[ScoredWindow] = []
    cell = float(stream.cell)

    for ws in enumerate_challenge_windows(stream.timestamps):
        we = int(ws) + WINDOW_US
        res = run_fn(stream, int(ws), we, conf_model, size_trees, always_emit=True)
        if res is None:
            continue
        gf = build_gate_features(res, stream, size_trees, reuse_geometry=True)
        gate = float(cv.score_gate_g1(gate_scaler, gate_clf, gf.reshape(1, -1))[0])
        row = emit_tii_row(res)
        base = float(res.confidence)
        cx_cells = float(res.cx) / cell
        cy_cells = float(res.cy) / cell
        event_count = float(len(res.current))
        event_rate_log = float(res.features[res.sel_idx, LOG_GLOBAL_IDX])
        temporal = state.features(
            cx_cells=cx_cells,
            cy_cells=cy_cells,
            gate_prob=gate,
            base_conf=base,
            event_rate_log=event_rate_log,
            event_count=event_count,
        )
        wgts = cv.window_gts_from_file(det_gts, int(ws), we)
        has_gt = len(wgts) > 0
        is_tp = cv.is_tp_box(int(ws), we, row[2], row[3], row[4], row[5], wgts) if has_gt else False
        top20 = _gt_near_any_candidate(res, wgts, cell) if has_gt else False
        out.append(
            ScoredWindow(
                sequence=sequence,
                ws=int(ws),
                we=we,
                row=row,
                base_conf=base,
                gate_score=gate,
                has_gt=has_gt,
                is_tp_if_emitted=is_tp,
                temporal=temporal,
                cx_cells=cx_cells,
                cy_cells=cy_cells,
                event_count=event_count,
                top20_has_tp_geometry=top20,
            )
        )
        state.push(
            HistorySlot(
                cx_cells=cx_cells,
                cy_cells=cy_cells,
                gate_prob=gate,
                base_conf=base,
                event_count=event_count,
            )
        )
    return out


def run_scored_sequences(
    sequences: list[str],
    split_dir: Path,
    conf_model,
    size_trees,
    gate_scaler,
    gate_clf,
) -> list[ScoredWindow]:
    rows: list[ScoredWindow] = []
    for sequence in sequences:
        print(f"    sequence {sequence}", flush=True)
        rows.extend(
            run_scored_sequence(
                sequence, split_dir, conf_model, size_trees, gate_scaler, gate_clf
            )
        )
    return rows


def champion_preds(windows: list[ScoredWindow], thr_d2: float) -> dict[str, list[tuple]]:
    preds: dict[str, list[tuple]] = defaultdict(list)
    for w in windows:
        if w.gate_score >= thr_d2:
            ws, we, cx, cy, ww, hh, _ = w.row
            preds[w.sequence].append((ws, we, cx, cy, ww, hh, float(w.gate_score)))
    return dict(preds)


def temporal_rescue_preds(
    windows: list[ScoredWindow],
    thr_d2: float,
    thr_rescue: float,
    rescue_scores: np.ndarray | None,
) -> dict[str, list[tuple]]:
    """D2 accepts unchanged; at most one rescue on rejected windows."""
    preds: dict[str, list[tuple]] = defaultdict(list)
    score_map = {}
    if rescue_scores is not None:
        for w, s in zip(windows, rescue_scores):
            score_map[(w.sequence, w.ws)] = float(s)
    for w in windows:
        ws, we, cx, cy, ww, hh, _ = w.row
        if w.gate_score >= thr_d2:
            preds[w.sequence].append((ws, we, cx, cy, ww, hh, float(w.gate_score)))
        else:
            rs = score_map.get((w.sequence, w.ws))
            if rs is not None and rs >= thr_rescue:
                preds[w.sequence].append((ws, we, cx, cy, ww, hh, float(rs)))
    return dict(preds)


def accepted_path_mismatches(
    champion: dict[str, list[tuple]],
    rescued: dict[str, list[tuple]],
    windows: list[ScoredWindow],
    thr_d2: float,
) -> int:
    """Count windows accepted by D2 whose TEMPORAL_RESCUE row differs."""
    champ_by_key = {}
    for seq, rows in champion.items():
        for r in rows:
            champ_by_key[(seq, int(r[0]), int(r[1]))] = r
    rescue_by_key = {}
    for seq, rows in rescued.items():
        for r in rows:
            rescue_by_key[(seq, int(r[0]), int(r[1]))] = r
    mismatches = 0
    for w in windows:
        if w.gate_score < thr_d2:
            continue
        key = (w.sequence, w.ws, w.we)
        a = champ_by_key.get(key)
        b = rescue_by_key.get(key)
        if a is None or b is None:
            mismatches += 1
            continue
        # start/end/cx/cy/width/height/confidence identical
        if a[:7] != b[:7]:
            mismatches += 1
    return mismatches


def empty_window_fpr(windows: list[ScoredWindow], preds: dict[str, list[tuple]]) -> float:
    pred_keys = {(seq, int(r[0]), int(r[1])) for seq, rows in preds.items() for r in rows}
    empty = 0
    empty_fp = 0
    for w in windows:
        if w.has_gt:
            continue
        empty += 1
        if (w.sequence, w.ws, w.we) in pred_keys:
            empty_fp += 1
    return empty_fp / max(empty, 1)


def score_bundle(preds: dict[str, list[tuple]], sequences: list[str], split_dir: Path) -> dict:
    return cv.score_preds(preds, sequences, split_dir)


def select_rescue_threshold(
    train_windows: list[ScoredWindow],
    thr_d2: float,
    oof_rescue: dict[tuple[str, int], float],
    split_dir: Path,
) -> float:
    """Maximize TII-style detection F1 of D2 accepts + OOF rescues on OUTER TRAIN."""
    rejected = [w for w in train_windows if w.gate_score < thr_d2]
    if not rejected:
        return 1.0
    scores = np.asarray([oof_rescue.get((w.sequence, w.ws), 0.0) for w in rejected], dtype=np.float64)
    qs = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    train_seqs = sorted({w.sequence for w in train_windows})
    best_t, best_f1 = 1.0, -1.0
    for t in qs:
        preds = temporal_rescue_preds(
            train_windows,
            thr_d2,
            float(t),
            rescue_scores=np.asarray(
                [oof_rescue.get((w.sequence, w.ws), -1.0) for w in train_windows],
                dtype=np.float64,
            ),
        )
        scored = score_bundle(preds, train_seqs, split_dir)
        f1 = float(scored["overall"].f1)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def inner_oof_rescue_probs(
    train_windows: list[ScoredWindow],
    thr_d2: float,
) -> dict[tuple[str, int], float]:
    rejected = [w for w in train_windows if w.gate_score < thr_d2]
    oof: dict[tuple[str, int], float] = {}
    if not rejected:
        return oof
    seqs = np.array(sorted({w.sequence for w in rejected}))
    if len(seqs) < 2:
        X = np.stack([w.temporal for w in rejected])
        y = np.asarray([1 if w.is_tp_if_emitted else 0 for w in rejected], dtype=np.int8)
        if y.min() == y.max():
            # degenerate: constant score
            for w in rejected:
                oof[(w.sequence, w.ws)] = float(y[0])
            return oof
        scaler, clf = fit_rescue(X, y)
        probs = score_rescue(scaler, clf, X)
        for w, p in zip(rejected, probs):
            oof[(w.sequence, w.ws)] = float(p)
        return oof

    by_seq: dict[str, list[ScoredWindow]] = defaultdict(list)
    for w in rejected:
        by_seq[w.sequence].append(w)

    kf = KFold(n_splits=min(5, len(seqs)), shuffle=True, random_state=42)
    for tr_i, va_i in kf.split(seqs):
        inner_train = set(seqs[tr_i].tolist())
        inner_val = set(seqs[va_i].tolist())
        tr_rows = [w for s in inner_train for w in by_seq[s]]
        va_rows = [w for s in inner_val for w in by_seq[s]]
        if not tr_rows or not va_rows:
            continue
        ytr = np.asarray([1 if w.is_tp_if_emitted else 0 for w in tr_rows], dtype=np.int8)
        if ytr.min() == ytr.max():
            for w in va_rows:
                oof[(w.sequence, w.ws)] = float(ytr[0])
            continue
        Xtr = np.stack([w.temporal for w in tr_rows])
        Xva = np.stack([w.temporal for w in va_rows])
        scaler, clf = fit_rescue(Xtr, ytr)
        probs = score_rescue(scaler, clf, Xva)
        for w, p in zip(va_rows, probs):
            oof[(w.sequence, w.ws)] = float(p)
    return oof


def metrics_row(fold_id: int, method: str, thr: float, scored: dict, empty_fpr: float, extra: dict | None = None) -> dict:
    ov = scored["overall"]
    row = {
        "fold": fold_id,
        "method": method,
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
        "empty_window_fpr": empty_fpr,
    }
    if extra:
        row.update(extra)
    return row


def write_pred_dirs(tmp: Path, sequences: list[str], split_dir: Path, preds: dict[str, list[tuple]]):
    gt_dir = tmp / "gt"
    pred_dir = tmp / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    for sequence in sequences:
        shutil.copy(
            split_dir / f"{sequence}_bb_windows_40ms.txt",
            gt_dir / f"{sequence}_bb_windows_40ms.txt",
        )
        write_tii_prediction_file(pred_dir / f"{sequence}_bb_windows_40ms.txt", preds.get(sequence, []))
    return gt_dir, pred_dir


def rescue_accounting(
    windows: list[ScoredWindow],
    thr_d2: float,
    thr_rescue: float,
    rescue_scores: np.ndarray,
) -> dict:
    rejected = 0
    attempts = 0
    emissions = 0
    rescue_tp = 0
    rescue_fp = 0
    top20_tp = 0
    for w, rs in zip(windows, rescue_scores):
        if w.gate_score >= thr_d2:
            continue
        rejected += 1
        attempts += 1
        if rs < thr_rescue:
            continue
        emissions += 1
        if w.is_tp_if_emitted:
            rescue_tp += 1
            if w.top20_has_tp_geometry:
                top20_tp += 1
        else:
            rescue_fp += 1
    return {
        "d2_rejected_windows": rejected,
        "rescue_attempts": attempts,
        "rescue_emissions": emissions,
        "rescue_true_positives": rescue_tp,
        "rescue_false_positives": rescue_fp,
        "rescue_precision": rescue_tp / max(emissions, 1),
        "new_tp_gained": rescue_tp,
        "new_fp_introduced": rescue_fp,
        "rescue_tp_top20_present": top20_tp,
        "rescue_tp_top1_geometry_ok": rescue_tp,  # by definition for rescue TP
    }


def ap_from_scores(windows: list[ScoredWindow], scores: np.ndarray, sequences: list[str], split_dir: Path) -> float:
    preds: dict[str, list[tuple]] = defaultdict(list)
    for w, s in zip(windows, scores):
        ws, we, cx, cy, ww, hh, _ = w.row
        preds[w.sequence].append((ws, we, cx, cy, ww, hh, float(s)))
    return float(score_bundle(dict(preds), sequences, split_dir)["overall"].ap50)


def benchmark_latency(
    sequence: str,
    split_dir: Path,
    conf_model,
    size_trees,
    gate_scaler,
    gate_clf,
    thr_d2: float,
    rescue_scaler,
    rescue_clf,
    thr_rescue: float,
    max_windows: int = 500,
) -> dict[str, list[float]]:
    stream = SequenceStream(sequence, split_dir)
    starts = list(enumerate_challenge_windows(stream.timestamps))[:max_windows]
    champ_ms: list[float] = []
    rescue_ms: list[float] = []
    state = CausalTemporalState(HISTORY)
    cell = float(stream.cell)

    for ws in starts:
        we = int(ws) + WINDOW_US
        # Champion D2 (gate path; geometry only if accepting — match deploy)
        t0 = perf_counter_ns()
        res = run_p1_window_fast(stream, int(ws), we, conf_model, size_trees, always_emit=True)
        if res is None:
            champ_ms.append((perf_counter_ns() - t0) / 1e6)
            rescue_ms.append(champ_ms[-1])
            continue
        gf = build_gate_features(res, stream, size_trees, reuse_geometry=True)
        gate = float(cv.score_gate_g1(gate_scaler, gate_clf, gf.reshape(1, -1))[0])
        accept = gate >= thr_d2
        t_champ = (perf_counter_ns() - t0) / 1e6
        champ_ms.append(t_champ)

        # Temporal rescue complete path (causal incremental state)
        t1 = perf_counter_ns()
        res2 = run_p1_window_fast(stream, int(ws), we, conf_model, size_trees, always_emit=True)
        if res2 is None:
            rescue_ms.append((perf_counter_ns() - t1) / 1e6)
            continue
        gf2 = build_gate_features(res2, stream, size_trees, reuse_geometry=True)
        gate2 = float(cv.score_gate_g1(gate_scaler, gate_clf, gf2.reshape(1, -1))[0])
        base = float(res2.confidence)
        cx_cells = float(res2.cx) / cell
        cy_cells = float(res2.cy) / cell
        event_count = float(len(res2.current))
        event_rate_log = float(res2.features[res2.sel_idx, LOG_GLOBAL_IDX])
        feats = state.features(
            cx_cells=cx_cells,
            cy_cells=cy_cells,
            gate_prob=gate2,
            base_conf=base,
            event_rate_log=event_rate_log,
            event_count=event_count,
        )
        if gate2 < thr_d2:
            rp = float(score_rescue(rescue_scaler, rescue_clf, feats.reshape(1, -1))[0])
            _ = rp >= thr_rescue
        state.push(
            HistorySlot(
                cx_cells=cx_cells,
                cy_cells=cy_cells,
                gate_prob=gate2,
                base_conf=base,
                event_count=event_count,
            )
        )
        rescue_ms.append((perf_counter_ns() - t1) / 1e6)
        _ = accept
    return {"CHAMPION_D2": champ_ms, "TEMPORAL_RESCUE": rescue_ms}


def fold_checkpoint_path(out_dir: Path, fold_id: int) -> Path:
    return out_dir / f"fold{fold_id}_done.json"


def run_fold(
    fold: dict,
    cache,
    table,
    split_dir: Path,
    out_dir: Path,
) -> dict:
    fold_id = int(fold["fold"])
    train_seqs = list(fold["train"])
    val_seqs = list(fold["validation"])
    print(f"Fold {fold_id}: D2 inner OOF...", flush=True)

    inner_val_recs, oof_g1, _oof_g2, train_gate_g1, _train_gate_g2 = cv.inner_oof_combined(
        train_seqs, cache, table, split_dir
    )
    thr_g1 = cv.detection_f1_threshold_from_records(inner_val_recs, oof_g1) if len(oof_g1) else 0.5

    train_idx = np.flatnonzero(np.isin(table["sequence"], train_seqs))
    mask = np.array([str(s) in set(train_seqs) for s in cache["sequence"]], dtype=bool)
    conf_model = cv.fit_confidence(cache["features"][mask], cache["target"][mask])
    size_trees = cv.fit_size_s2(table, train_idx, split_dir)

    Xg = np.stack([r.gate_features for r in train_gate_g1])
    yg = np.array([r.is_tp_if_emitted for r in train_gate_g1], dtype=np.int8)
    g1_scaler, g1_clf = cv.fit_gate_g1(Xg, yg)

    print(f"Fold {fold_id}: scored OUTER TRAIN (temporal)...", flush=True)
    train_windows = run_scored_sequences(
        train_seqs, split_dir, conf_model, size_trees, g1_scaler, g1_clf
    )
    print(f"Fold {fold_id}: scored OUTER VAL (temporal)...", flush=True)
    val_windows = run_scored_sequences(
        val_seqs, split_dir, conf_model, size_trees, g1_scaler, g1_clf
    )

    print(f"Fold {fold_id}: rescue INNER-OOF + threshold...", flush=True)
    oof_rescue = inner_oof_rescue_probs(train_windows, thr_g1)
    thr_rescue = select_rescue_threshold(train_windows, thr_g1, oof_rescue, split_dir)

    rejected_train = [w for w in train_windows if w.gate_score < thr_g1]
    if rejected_train:
        ytr = np.asarray([1 if w.is_tp_if_emitted else 0 for w in rejected_train], dtype=np.int8)
        if ytr.min() == ytr.max():
            # Still fit for a defined model; LR needs both classes — fallback constant.
            rescue_scaler, rescue_clf = None, None
            constant_prob = float(ytr[0])
        else:
            Xtr = np.stack([w.temporal for w in rejected_train])
            rescue_scaler, rescue_clf = fit_rescue(Xtr, ytr)
            constant_prob = None
    else:
        rescue_scaler, rescue_clf, constant_prob = None, None, 0.0

    def rescue_probs_for(windows: list[ScoredWindow]) -> np.ndarray:
        out = np.full(len(windows), -1.0, dtype=np.float64)
        for i, w in enumerate(windows):
            if w.gate_score >= thr_g1:
                continue
            if rescue_scaler is None:
                out[i] = float(constant_prob if constant_prob is not None else 0.0)
            else:
                out[i] = float(score_rescue(rescue_scaler, rescue_clf, w.temporal.reshape(1, -1))[0])
        return out

    val_rescue_scores = rescue_probs_for(val_windows)
    champ_preds = champion_preds(val_windows, thr_g1)
    rescue_preds = temporal_rescue_preds(val_windows, thr_g1, thr_rescue, val_rescue_scores)
    mismatches = accepted_path_mismatches(champ_preds, rescue_preds, val_windows, thr_g1)

    champ_scored = score_bundle(champ_preds, val_seqs, split_dir)
    rescue_scored = score_bundle(rescue_preds, val_seqs, split_dir)
    champ_fpr = empty_window_fpr(val_windows, champ_preds)
    rescue_fpr = empty_window_fpr(val_windows, rescue_preds)
    acct = rescue_accounting(val_windows, thr_g1, thr_rescue, val_rescue_scores)

    # AP side diagnostic (unthresholded)
    base_ap = ap_from_scores(
        val_windows, np.asarray([w.base_conf for w in val_windows]), val_seqs, split_dir
    )
    gate_ap = ap_from_scores(
        val_windows, np.asarray([w.gate_score for w in val_windows]), val_seqs, split_dir
    )

    # Compact unthresholded cache
    cache_rows = []
    for w, rs in zip(val_windows, val_rescue_scores):
        cache_rows.append(
            {
                "fold": fold_id,
                "sequence": w.sequence,
                "window_start_us": w.ws,
                "window_end_us": w.we,
                "cx": w.row[2],
                "cy": w.row[3],
                "width": w.row[4],
                "height": w.row[5],
                "base_confidence": w.base_conf,
                "d2_gate_probability": w.gate_score,
                "temporal_rescue_probability": rs if w.gate_score < thr_g1 else "",
                "d2_accepted": int(w.gate_score >= thr_g1),
                "is_tp_if_emitted": int(w.is_tp_if_emitted),
                "has_gt": int(w.has_gt),
            }
        )
    cv.write_csv(out_dir / f"unthresholded_fold{fold_id}.csv", cache_rows)

    # TII parity for both methods
    for method, preds in [("CHAMPION_D2", champ_preds), ("TEMPORAL_RESCUE", rescue_preds)]:
        with tempfile.TemporaryDirectory() as tmp:
            gt_dir, pred_dir = write_pred_dirs(Path(tmp), val_seqs, split_dir, preds)
            local = aggregate_tii(evaluate_dirs_tii(gt_dir, pred_dir))
            excel = out_dir / f"tii_fold{fold_id}_{method}.xlsx"
            tii = cv.run_tii_official(gt_dir, pred_dir, excel)
            cv.assert_tii_parity(local, tii, fold_id, method)

    # Latency (Part H) — use this fold's models; accumulate later
    lat_samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # Constant-prob fallback: still exercise temporal feature path with a dummy LR
    if rescue_scaler is None:
        X_dummy = np.zeros((2, N_TEMPORAL_FEATURES), dtype=np.float64)
        y_dummy = np.asarray([0, 1], dtype=np.int8)
        rescue_scaler, rescue_clf = fit_rescue(X_dummy, y_dummy)
    for seq in LATENCY_SEQS:
        if not (split_dir / f"{seq}_labeled_events.npy").exists():
            continue
        print(f"Fold {fold_id}: latency {seq}", flush=True)
        samples = benchmark_latency(
            seq,
            split_dir,
            conf_model,
            size_trees,
            g1_scaler,
            g1_clf,
            thr_g1,
            rescue_scaler,
            rescue_clf,
            thr_rescue,
        )
        sensor = cv.sensor_name(seq)
        for method, vals in samples.items():
            lat_samples[method][sensor].extend(vals)
            lat_samples[method]["ALL"].extend(vals)

    # Per-sequence rows
    seq_rows = []
    for method, preds, scored in [
        ("CHAMPION_D2", champ_preds, champ_scored),
        ("TEMPORAL_RESCUE", rescue_preds, rescue_scored),
    ]:
        for sequence, m in scored["per_seq"].items():
            seq_rows.append(
                {
                    "fold": fold_id,
                    "method": method,
                    "sequence": sequence,
                    "sensor": cv.sensor_name(sequence),
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "ap50": m.ap50,
                    "tp": m.tp,
                    "fp": m.fp,
                    "fn": m.fn,
                    "mean_matched_iou": m.mean_matched_iou,
                }
            )

    fold_rows = [
        metrics_row(fold_id, "CHAMPION_D2", thr_g1, champ_scored, champ_fpr, {"rescue_threshold": ""}),
        metrics_row(
            fold_id,
            "TEMPORAL_RESCUE",
            thr_rescue,
            rescue_scored,
            rescue_fpr,
            {
                "d2_threshold": thr_g1,
                "accepted_path_mismatches": mismatches,
                **{f"acct_{k}": v for k, v in acct.items()},
                "ap50_base_conf": base_ap,
                "ap50_d2_gate": gate_ap,
            },
        ),
    ]

    result = {
        "fold": fold_id,
        "thr_d2": thr_g1,
        "thr_rescue": thr_rescue,
        "mismatches": mismatches,
        "fold_rows": fold_rows,
        "seq_rows": seq_rows,
        "accounting": acct,
        "latency": {m: {s: v for s, v in sensors.items()} for m, sensors in lat_samples.items()},
        "champ_overall": {
            "precision": champ_scored["overall"].precision,
            "recall": champ_scored["overall"].recall,
            "f1": champ_scored["overall"].f1,
            "ap50": champ_scored["overall"].ap50,
            "tp": champ_scored["overall"].tp,
            "fp": champ_scored["overall"].fp,
            "fn": champ_scored["overall"].fn,
            "mean_matched_iou": champ_scored["overall"].mean_matched_iou,
            "sequence_macro_f1": champ_scored["sequence_macro_f1"],
            "sequence_macro_recall": champ_scored["sequence_macro_recall"],
            "empty_window_fpr": champ_fpr,
        },
        "rescue_overall": {
            "precision": rescue_scored["overall"].precision,
            "recall": rescue_scored["overall"].recall,
            "f1": rescue_scored["overall"].f1,
            "ap50": rescue_scored["overall"].ap50,
            "tp": rescue_scored["overall"].tp,
            "fp": rescue_scored["overall"].fp,
            "fn": rescue_scored["overall"].fn,
            "mean_matched_iou": rescue_scored["overall"].mean_matched_iou,
            "sequence_macro_f1": rescue_scored["sequence_macro_f1"],
            "sequence_macro_recall": rescue_scored["sequence_macro_recall"],
            "empty_window_fpr": rescue_fpr,
        },
        "n_train_rejected": len(rejected_train),
        "n_val_windows": len(val_windows),
    }
    write_atomic_json(fold_checkpoint_path(out_dir, fold_id), {
        "fold": fold_id,
        "done": True,
        "thr_d2": thr_g1,
        "thr_rescue": thr_rescue,
        "mismatches": mismatches,
        "champ": result["champ_overall"],
        "rescue": result["rescue_overall"],
        "accounting": acct,
    })
    return result


def pool_metrics(fold_results: list[dict], key: str) -> dict:
    """Micro-pool TP/FP/FN across folds; recompute precision/recall/F1."""
    tp = sum(int(r[key]["tp"]) for r in fold_results)
    fp = sum(int(r[key]["fp"]) for r in fold_results)
    fn = sum(int(r[key]["fn"]) for r in fold_results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Weighted mean matched IoU / macro averages across folds by TP or equal weight
    ious = [r[key]["mean_matched_iou"] for r in fold_results]
    empty = [r[key]["empty_window_fpr"] for r in fold_results]
    macro_f1 = float(np.mean([r[key]["sequence_macro_f1"] for r in fold_results]))
    macro_rec = float(np.mean([r[key]["sequence_macro_recall"] for r in fold_results]))
    aps = [r[key]["ap50"] for r in fold_results]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": float(np.mean(aps)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_matched_iou": float(np.average(ious, weights=[max(r[key]["tp"], 1) for r in fold_results])),
        "sequence_macro_f1": macro_f1,
        "sequence_macro_recall": macro_rec,
        "empty_window_fpr": float(np.mean(empty)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=SPLIT)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/all_window_candidates.npz"))
    parser.add_argument("--table", type=Path, default=Path("artifacts/candidate_table.csv"))
    parser.add_argument("--folds", type=Path, default=Path("sequence_folds.json"))
    parser.add_argument("--fold-ids", type=str, default="1,4")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/runs/2026-09-05/temporal_rescue_screen"),
    )
    args = parser.parse_args()
    if args.no_resume:
        args.resume = False
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Safety: never touch Testing_sets
    if "Testing_sets" in str(args.split_dir):
        raise SystemExit("Refusing Testing_sets path")

    cache = cv.load_cache(args.cache)
    table = cv.load_table(args.table)
    folds_all = json.loads(args.folds.read_text(encoding="utf-8"))
    fold_filter = parse_fold_ids(args.fold_ids, n_folds=len(folds_all))
    if fold_filter is None:
        fold_filter = SCREEN_FOLDS
    folds = [f for f in folds_all if int(f["fold"]) in set(fold_filter)]
    print(f"SCREEN fold_ids={sorted(int(f['fold']) for f in folds)}", flush=True)

    fold_results: list[dict] = []
    compare_fold_rows: list[dict] = []
    compare_seq_rows: list[dict] = []
    latency_all: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    # Reload prior fold CSV rows when resuming
    if args.resume:
        compare_fold_rows = cv.read_csv_rows(args.out_dir / "compare_by_fold.csv")
        compare_seq_rows = cv.read_csv_rows(args.out_dir / "compare_by_sequence.csv")

    done_folds = set()
    for fold in folds:
        fold_id = int(fold["fold"])
        ckpt = fold_checkpoint_path(args.out_dir, fold_id)
        if args.resume and ckpt.exists():
            print(f"Fold {fold_id}: resume from {ckpt}", flush=True)
            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            lat = cv.load_latency_fold(args.out_dir, fold_id)
            for method, sensors in lat.items():
                for sensor, vals in sensors.items():
                    latency_all[method][sensor].extend(vals)
            fold_results.append(
                {
                    "fold": fold_id,
                    "thr_d2": payload["thr_d2"],
                    "thr_rescue": payload["thr_rescue"],
                    "mismatches": payload["mismatches"],
                    "champ_overall": payload["champ"],
                    "rescue_overall": payload["rescue"],
                    "accounting": payload["accounting"],
                }
            )
            done_folds.add(fold_id)
            continue

        # Drop stale CSV rows for this fold before rewrite
        compare_fold_rows = [r for r in compare_fold_rows if int(float(r["fold"])) != fold_id]
        compare_seq_rows = [r for r in compare_seq_rows if int(float(r["fold"])) != fold_id]

        result = run_fold(fold, cache, table, args.split_dir, args.out_dir)
        fold_results.append(result)
        compare_fold_rows.extend(result["fold_rows"])
        compare_seq_rows.extend(result["seq_rows"])
        cv.save_latency_fold(args.out_dir, fold_id, result["latency"])
        for method, sensors in result["latency"].items():
            for sensor, vals in sensors.items():
                latency_all[method][sensor].extend(vals)
        write_csv_union(args.out_dir / "compare_by_fold.csv", compare_fold_rows)
        write_csv_union(args.out_dir / "compare_by_sequence.csv", compare_seq_rows)
        done_folds.add(fold_id)

    write_csv_union(args.out_dir / "compare_by_fold.csv", compare_fold_rows)
    write_csv_union(args.out_dir / "compare_by_sequence.csv", compare_seq_rows)

    pooled_d2 = pool_metrics(fold_results, "champ_overall")
    pooled_tr = pool_metrics(fold_results, "rescue_overall")
    mismatches_total = sum(int(r["mismatches"]) for r in fold_results)

    acct_pool = {
        "d2_rejected_windows": sum(int(r["accounting"]["d2_rejected_windows"]) for r in fold_results),
        "rescue_attempts": sum(int(r["accounting"]["rescue_attempts"]) for r in fold_results),
        "rescue_emissions": sum(int(r["accounting"]["rescue_emissions"]) for r in fold_results),
        "rescue_true_positives": sum(int(r["accounting"]["rescue_true_positives"]) for r in fold_results),
        "rescue_false_positives": sum(int(r["accounting"]["rescue_false_positives"]) for r in fold_results),
        "new_tp_gained": sum(int(r["accounting"]["new_tp_gained"]) for r in fold_results),
        "new_fp_introduced": sum(int(r["accounting"]["new_fp_introduced"]) for r in fold_results),
        "rescue_tp_top20_present": sum(int(r["accounting"]["rescue_tp_top20_present"]) for r in fold_results),
        "rescue_tp_top1_geometry_ok": sum(int(r["accounting"]["rescue_tp_top1_geometry_ok"]) for r in fold_results),
    }
    acct_pool["rescue_precision"] = acct_pool["rescue_true_positives"] / max(
        acct_pool["rescue_emissions"], 1
    )

    # Latency percentiles
    latency_rows = []
    temporal_p95_all = None
    for method, sensors in latency_all.items():
        for sensor, vals in sensors.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            pct = {
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
            latency_rows.append(
                {
                    "method": method,
                    "sensor": sensor,
                    "n": int(arr.size),
                    "p50_ms": pct["p50"],
                    "p95_ms": pct["p95"],
                    "p99_ms": pct["p99"],
                }
            )
            if method == "TEMPORAL_RESCUE" and sensor == "ALL":
                temporal_p95_all = pct["p95"]
    cv.write_csv(args.out_dir / "latency_screen.csv", latency_rows)

    # Gate-score AP cache summary (folds 1+4)
    ap_rows = []
    for r in fold_results:
        # from TEMPORAL_RESCUE fold_rows extras if present in CSV
        pass
    # Re-read from compare_by_fold extras
    for r in compare_fold_rows:
        if r.get("method") == "TEMPORAL_RESCUE" and "ap50_base_conf" in r and r["ap50_base_conf"] != "":
            ap_rows.append(
                {
                    "fold": int(float(r["fold"])),
                    "ap50_base_conf": float(r["ap50_base_conf"]),
                    "ap50_d2_gate": float(r["ap50_d2_gate"]),
                }
            )
    if ap_rows:
        cv.write_csv(args.out_dir / "gate_score_ap_diagnostic.csv", ap_rows)

    pooled_rows = [
        {"method": "CHAMPION_D2", **pooled_d2, "accepted_path_mismatches": ""},
        {"method": "TEMPORAL_RESCUE", **pooled_tr, "accepted_path_mismatches": mismatches_total},
    ]
    write_csv_union(args.out_dir / "pooled_screen_1_4.csv", pooled_rows)
    write_csv_union(args.out_dir / "rescue_accounting.csv", [{**acct_pool}])

    # SCREEN promotion criteria (frozen; no post-hoc changes)
    screen_a = pooled_tr["f1"] >= pooled_d2["f1"] + 0.03
    screen_b = pooled_tr["recall"] >= pooled_d2["recall"] + 0.04
    screen_c = pooled_tr["precision"] >= 0.50
    screen_d = pooled_tr["empty_window_fpr"] <= 0.02
    screen_e = mismatches_total == 0
    screen_f = (temporal_p95_all is not None) and (temporal_p95_all <= 40.0)
    all_pass = all([screen_a, screen_b, screen_c, screen_d, screen_e, screen_f])
    full_cv = bool(all_pass)

    criteria = {
        "SCREEN_A_f1": {"pass": screen_a, "temporal": pooled_tr["f1"], "d2": pooled_d2["f1"]},
        "SCREEN_B_recall": {"pass": screen_b, "temporal": pooled_tr["recall"], "d2": pooled_d2["recall"]},
        "SCREEN_C_precision": {"pass": screen_c, "temporal": pooled_tr["precision"]},
        "SCREEN_D_empty_fpr": {"pass": screen_d, "temporal": pooled_tr["empty_window_fpr"]},
        "SCREEN_E_invariance": {"pass": screen_e, "accepted_path_mismatches": mismatches_total},
        "SCREEN_F_latency_p95": {"pass": screen_f, "temporal_p95_ms": temporal_p95_all},
        "FULL_CV_RECOMMENDED": full_cv,
    }
    write_atomic_json(args.out_dir / "screen_criteria.json", criteria)
    write_atomic_text(
        args.out_dir / "FULL_CV_RECOMMENDED.txt",
        f"FULL_CV_RECOMMENDED={'True' if full_cv else 'False'}\n",
    )

    print(json.dumps(criteria, indent=2), flush=True)
    print(f"FULL_CV_RECOMMENDED={full_cv}", flush=True)


if __name__ == "__main__":
    main()
