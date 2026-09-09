from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

from orbitsight.io import Detection, read_detection_file


@dataclass(frozen=True)
class EvaluationResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    ap50: float
    mean_matched_iou: float
    num_gt: int
    num_predictions: int


def temporal_overlap(a: Detection, b: Detection) -> bool:
    return max(a.start_us, b.start_us) < min(a.end_us, b.end_us)


def iou(a: Detection, b: Detection) -> float:
    ax1, ay1 = a.cx - a.width / 2.0, a.cy - a.height / 2.0
    ax2, ay2 = a.cx + a.width / 2.0, a.cy + a.height / 2.0
    bx1, by1 = b.cx - b.width / 2.0, b.cy - b.height / 2.0
    bx2, by2 = b.cx + b.width / 2.0, b.cy + b.height / 2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0.0 else 0.0


def _ap_from_ranked(tp_flags: np.ndarray, fp_flags: np.ndarray, num_gt: int) -> float:
    if num_gt <= 0 or tp_flags.size == 0:
        return 0.0
    tp_cum = np.cumsum(tp_flags, dtype=np.float64)
    fp_cum = np.cumsum(fp_flags, dtype=np.float64)
    recalls = tp_cum / float(num_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def _evaluate_ranked(gt_by_sequence: Mapping[str, Sequence[Detection]], predictions: Sequence[tuple[str, Detection]], iou_threshold: float) -> EvaluationResult:
    matched = {name: np.zeros(len(gt), dtype=bool) for name, gt in gt_by_sequence.items()}
    ranked = sorted(predictions, key=lambda item: item[1].confidence, reverse=True)
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    matched_ious: list[float] = []
    for sequence, pred in ranked:
        gt = gt_by_sequence.get(sequence, ())
        best_idx = -1
        best_iou = -1.0
        for idx, truth in enumerate(gt):
            if matched[sequence][idx] or not temporal_overlap(pred, truth):
                continue
            overlap = iou(pred, truth)
            if overlap >= iou_threshold and overlap > best_iou:
                best_idx, best_iou = idx, overlap
        if best_idx >= 0:
            matched[sequence][best_idx] = True
            tp_flags.append(1)
            fp_flags.append(0)
            matched_ious.append(best_iou)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    num_gt = sum(len(v) for v in gt_by_sequence.values())
    tp = int(sum(tp_flags))
    fp = int(sum(fp_flags))
    fn = int(num_gt - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / num_gt if num_gt else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap50 = _ap_from_ranked(np.asarray(tp_flags), np.asarray(fp_flags), num_gt)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return EvaluationResult(tp, fp, fn, precision, recall, f1, ap50, mean_iou, num_gt, len(ranked))


def evaluate_sequence(gt: Sequence[Detection], predictions: Sequence[Detection], iou_threshold: float = 0.5) -> EvaluationResult:
    return _evaluate_ranked({"sequence": gt}, [("sequence", p) for p in predictions], iou_threshold)


def _sequence_name_from_gt(path: Path) -> str:
    suffix = "_bb_windows_40ms.txt"
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem


def _prediction_path(pred_dir: Path, sequence: str) -> Path | None:
    candidates = [pred_dir / f"{sequence}_pred.txt", pred_dir / f"{sequence}.txt"]
    return next((p for p in candidates if p.exists()), None)


def evaluate_dataset(gt_dir: str | Path, prediction_dir: str | Path, iou_threshold: float = 0.5) -> EvaluationResult:
    """Evaluate using documented temporal-overlap + IoU matching.

    Before treating results as authoritative, compare against TII's supplied
    OrbitSight_DataLoader/evaluate.py on identical prediction files.
    """
    gt_dir, prediction_dir = Path(gt_dir), Path(prediction_dir)
    gt_by_sequence: dict[str, list[Detection]] = {}
    predictions: list[tuple[str, Detection]] = []
    for gt_path in sorted(gt_dir.glob("*_bb_windows_40ms.txt")):
        sequence = _sequence_name_from_gt(gt_path)
        gt_by_sequence[sequence] = read_detection_file(gt_path)
        pred_path = _prediction_path(prediction_dir, sequence)
        if pred_path is not None:
            predictions.extend((sequence, p) for p in read_detection_file(pred_path))
    if not gt_by_sequence:
        raise FileNotFoundError(f"No *_bb_windows_40ms.txt files found in {gt_dir}")
    return _evaluate_ranked(gt_by_sequence, predictions, iou_threshold)
