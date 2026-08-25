from orbitsight.evaluation.metrics import evaluate_sequence, iou
from orbitsight.io import Detection


def box(cx=10, cy=10, confidence=1.0, start=0, end=40_000):
    return Detection(start, end, cx, cy, 10, 10, confidence)


def test_iou_identical_is_one():
    assert iou(box(), box()) == 1.0


def test_one_to_one_matching_penalizes_duplicate_prediction():
    gt = [box()]
    preds = [box(confidence=0.9), box(confidence=0.8)]
    result = evaluate_sequence(gt, preds)
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.precision == 0.5
    assert result.recall == 1.0


def test_non_overlapping_time_is_false_positive():
    gt = [box(start=0, end=40_000)]
    pred = box(start=40_000, end=80_000)
    result = evaluate_sequence(gt, [pred])
    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1
