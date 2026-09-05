from __future__ import annotations

"""TII OrbitSight_DataLoader/evaluate.py matching (bit-compatible IoU)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def cx_cy_wh_to_xyxy(cx, cy, w, h):
    x1 = cx - (w - 1) / 2
    y1 = cy - (h - 1) / 2
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    return x1, y1, x2, y2


def tii_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = cx_cy_wh_to_xyxy(*box_a)
    bx1, by1, bx2, by2 = cx_cy_wh_to_xyxy(*box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def windows_overlap(ws_a, we_a, ws_b, we_b) -> bool:
    return ws_a < we_b and we_a > ws_b


@dataclass
class TIIMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    ap50: float
    mean_matched_iou: float
    sequence_macro_matched_iou: float
    n_gt: int
    n_pred: int


def compute_ap(tp_flags: list[int], fp_flags: list[int], n_gt: int) -> float:
    """Match OrbitSight_DataLoader/evaluate.py compute_ap exactly."""
    if n_gt == 0:
        return float("nan")
    if not tp_flags:
        return 0.0
    tp = np.asarray(tp_flags, dtype=np.float64)
    fp = np.asarray(fp_flags, dtype=np.float64)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / float(n_gt)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)
    recalls = np.concatenate([[0.0], recalls, [recalls[-1] if len(recalls) else 0.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def match_and_score(
    gt_rows: list[tuple],
    pred_rows: list[tuple],
    iou_thresh: float = 0.5,
) -> TIIMetrics:
    """Mirror TII match_predictions + compute_ap / compute_prf1."""
    n_gt = len(gt_rows)
    ranked = sorted(pred_rows, key=lambda r: r[6], reverse=True)
    if n_gt == 0 and len(ranked) == 0:
        return TIIMetrics(
            tp=0, fp=0, fn=0, precision=0.0, recall=0.0, f1=0.0,
            ap50=float("nan"), mean_matched_iou=0.0, sequence_macro_matched_iou=0.0,
            n_gt=0, n_pred=0,
        )
    gt_matched = [False] * n_gt
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    matched_ious: list[float] = []
    for pred in ranked:
        best_iou, best_idx = 0.0, -1
        for gi, gt in enumerate(gt_rows):
            if gt_matched[gi]:
                continue
            if not windows_overlap(pred[0], pred[1], gt[0], gt[1]):
                continue
            score = tii_iou(pred[2:6], gt[2:6])
            if score > best_iou:
                best_iou, best_idx = score, gi
        if best_iou >= iou_thresh and best_idx >= 0:
            tp_flags.append(1)
            fp_flags.append(0)
            gt_matched[best_idx] = True
            matched_ious.append(best_iou)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    tp = int(sum(tp_flags))
    fp = int(sum(fp_flags))
    fn = int(n_gt - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap = compute_ap(tp_flags, fp_flags, n_gt)
    seq_mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return TIIMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        ap50=float(ap),
        mean_matched_iou=seq_mean_iou,
        sequence_macro_matched_iou=seq_mean_iou,
        n_gt=n_gt,
        n_pred=len(ranked),
    )


def write_tii_prediction_file(path: Path, rows: list[tuple]) -> None:
    """Write TII evaluate.py-compatible predictions (integer boxes as TII load_pred requires)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "window_start_timestamp_us\twindow_end_timestamp_us\t"
            "center_x\tcenter_y\twidth\theight\tconfidence\n"
        )
        for ws, we, cx, cy, w, h, conf in rows:
            handle.write(
                f"{int(ws)}\t{int(we)}\t{int(round(cx))}\t{int(round(cy))}\t"
                f"{int(round(w))}\t{int(round(h))}\t{float(conf)}\n"
            )


def load_tii_gt(path: Path) -> list[tuple]:
    import csv

    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                (
                    int(row["window_start_timestamp_us"]),
                    int(row["window_end_timestamp_us"]),
                    int(row["center_x"]),
                    int(row["center_y"]),
                    int(row["width"]),
                    int(row["height"]),
                )
            )
    return rows


def load_tii_pred(path: Path) -> list[tuple]:
    import csv

    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            conf = float(row["confidence"]) if "confidence" in row and row["confidence"] else 1.0
            rows.append(
                (
                    int(row["window_start_timestamp_us"]),
                    int(row["window_end_timestamp_us"]),
                    int(row["center_x"]),
                    int(row["center_y"]),
                    int(row["width"]),
                    int(row["height"]),
                    conf,
                )
            )
    return rows


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose threshold maximizing F1 on provided scores (candidate-label or detection)."""
    if len(y_true) == 0:
        return 0.5
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


def pooled_percentiles(samples: dict[str, list[float]]) -> dict[str, float]:
    """Pooled p50/p95/p99 across all benchmark samples (not mean of fold percentiles)."""
    all_vals = [v for vals in samples.values() for v in vals]
    if not all_vals:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "p99_ms": float("nan"), "n": 0}
    arr = np.asarray(all_vals, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "n": int(arr.size),
    }


def select_detection_f1_threshold(
    window_scores: np.ndarray,
    window_emit_mask: np.ndarray,
    window_is_tp_if_emitted: np.ndarray,
    window_has_gt: np.ndarray,
) -> float:
    """
    Sweep confidence/gate scores on pooled windows maximizing TII-style detection F1.

    window_emit_mask: True where a Top-1 box exists (unthresholded P1 ran).
    window_is_tp_if_emitted: True if emitted box would match GT at IoU>=0.5.
    window_has_gt: True if window has any GT box.
    """
    scores = window_scores[window_emit_mask]
    is_tp = window_is_tp_if_emitted[window_emit_mask]
    has_gt = window_has_gt[window_emit_mask]
    if len(scores) == 0:
        return 0.5
    n_gt = int(np.sum(has_gt))
    if n_gt == 0:
        return float(np.max(scores)) + 1.0
    qs = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_t, best_f1 = 0.5, -1.0
    for t in qs:
        emit = scores >= t
        tp = int(np.sum(emit & is_tp))
        fp = int(np.sum(emit & ~is_tp))
        fn = n_gt - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def preds_from_thresholded_windows(
    all_rows: list[tuple],
    scores: np.ndarray,
    threshold: float,
) -> list[tuple]:
    """Filter unthresholded window rows (ws, we, cx, cy, w, h, score) by score >= threshold."""
    out = []
    for row, s in zip(all_rows, scores):
        if float(s) >= threshold:
            ws, we, cx, cy, w, h, _ = row
            out.append((ws, we, cx, cy, w, h, float(s)))
    return out


def evaluate_dirs_tii(gt_dir: Path, pred_dir: Path, iou_thresh: float = 0.5) -> dict[str, TIIMetrics]:
    out: dict[str, TIIMetrics] = {}
    for gt_path in sorted(gt_dir.glob("*_bb_windows_40ms.txt")):
        pred_path = pred_dir / gt_path.name
        gt = load_tii_gt(gt_path)
        pred = load_tii_pred(pred_path) if pred_path.exists() else []
        out[gt_path.name.replace("_bb_windows_40ms.txt", "")] = match_and_score(gt, pred, iou_thresh)
    return out


def aggregate_tii(per_seq: dict[str, TIIMetrics]) -> TIIMetrics:
    tp = sum(m.tp for m in per_seq.values())
    fp = sum(m.fp for m in per_seq.values())
    fn = sum(m.fn for m in per_seq.values())
    n_gt = sum(m.n_gt for m in per_seq.values())
    n_pred = sum(m.n_pred for m in per_seq.values())
    # Pool AP: recompute would need all preds; use micro F1 and mean AP of sequences for map
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    aps = [
        m.ap50
        for m in per_seq.values()
        if m.ap50 is not None and not (isinstance(m.ap50, float) and np.isnan(m.ap50))
    ]
    seq_macro_ious = [m.mean_matched_iou for m in per_seq.values() if m.tp > 0]
    weighted_iou = sum(m.mean_matched_iou * m.tp for m in per_seq.values() if m.tp > 0)
    tp_for_iou = sum(m.tp for m in per_seq.values())
    return TIIMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        ap50=float(np.mean(aps)) if aps else float("nan"),
        mean_matched_iou=weighted_iou / tp_for_iou if tp_for_iou > 0 else 0.0,
        sequence_macro_matched_iou=float(np.mean(seq_macro_ious)) if seq_macro_ious else 0.0,
        n_gt=n_gt,
        n_pred=n_pred,
    )
