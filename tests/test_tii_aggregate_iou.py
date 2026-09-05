"""Unit tests for TII aggregate mean_matched_iou weighting."""

from orbitsight.evaluation.tii_style import aggregate_tii, TIIMetrics


def test_pooled_mean_matched_iou_weighted_by_tp():
    per_seq = {
        "A": TIIMetrics(
            tp=1, fp=0, fn=0,
            precision=1.0, recall=1.0, f1=1.0, ap50=1.0,
            mean_matched_iou=1.0, sequence_macro_matched_iou=1.0,
            n_gt=1, n_pred=1,
        ),
        "B": TIIMetrics(
            tp=9, fp=0, fn=0,
            precision=1.0, recall=1.0, f1=1.0, ap50=0.5,
            mean_matched_iou=0.5, sequence_macro_matched_iou=0.5,
            n_gt=9, n_pred=9,
        ),
    }
    agg = aggregate_tii(per_seq)
    assert abs(agg.mean_matched_iou - 0.55) < 1e-9
    assert abs(agg.sequence_macro_matched_iou - 0.75) < 1e-9
    assert agg.mean_matched_iou != agg.sequence_macro_matched_iou
