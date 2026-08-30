import math

from orbitsight.evaluation.detection_aggregate import aggregate_detection_metrics, failure_buckets


def _detail(iou: float, sequence: str, fold: int = 0, proposal_hit: bool = True, ranker_hit: bool = True) -> dict:
    oracle_iou = min(1.0, iou + 0.2) if ranker_hit else 0.0
    return {
        "fold": fold,
        "sequence": sequence,
        "proposal_hit": proposal_hit,
        "ranker_hit": ranker_hit,
        "iou": iou,
        "oracle_iou": oracle_iou if ranker_hit else 0.0,
        "iou50": float(iou >= 0.5),
        "iou75": float(iou >= 0.75),
        "centre_error": 1.0,
    }


def test_pooled_micro_counts_all_gt_not_mean_of_fold_pcts():
    # Fold 0: 1/2 success; fold 1: 1/100 success -> pooled micro = 2/102, not mean(50%, 1%)
    details = [
        _detail(1.0, "seq_a", fold=0),
        _detail(0.0, "seq_a", fold=0),
        *[_detail(0.0, "seq_b", fold=1) for _ in range(99)],
        _detail(1.0, "seq_b", fold=1),
    ]
    agg = aggregate_detection_metrics(details)
    assert agg["n_gt"] == 102.0
    assert abs(agg["pooled_micro_iou50_pct"] - 100.0 * 2 / 102) < 1e-9
    assert abs(agg["fold_mean_iou50_pct"] - 100.0 * (0.5 + 0.01) / 2) < 1e-9


def test_sequence_macro_unweighted_over_sequences():
    details = [
        _detail(1.0, "seq_a"),
        _detail(1.0, "seq_a"),
        _detail(0.0, "seq_b"),
        _detail(0.0, "seq_b"),
        _detail(0.0, "seq_b"),
        _detail(0.0, "seq_b"),
    ]
    agg = aggregate_detection_metrics(details)
    assert abs(agg["sequence_macro_iou50_pct"] - 100.0 * (1.0 + 0.0) / 2) < 1e-9


def test_failure_buckets_success_not_counted_as_failure():
    details = [
        {**_detail(0.9, "s"), "proposal_hit": True, "ranker_hit": True, "oracle_iou": 0.9},
        {**_detail(0.3, "s"), "proposal_hit": False, "ranker_hit": False, "oracle_iou": 0.0},
        {**_detail(0.2, "s"), "proposal_hit": True, "ranker_hit": False, "oracle_iou": 0.0},
    ]
    rows = {r["bucket"]: r for r in failure_buckets(details)}
    assert rows["success_iou50"]["count"] == 1
    assert rows["proposal_miss"]["count"] == 1
    assert rows["ranking_error"]["count"] == 1
    assert rows["success_iou50"]["pct_failures_only"] == 0.0
    assert abs(rows["proposal_miss"]["pct_failures_only"] - 50.0) < 1e-9


def test_failure_bucket_centre_vs_size_split():
    details = [
        {**_detail(0.3, "s"), "proposal_hit": True, "ranker_hit": True, "oracle_iou": 0.3},
        {**_detail(0.4, "s"), "proposal_hit": True, "ranker_hit": True, "oracle_iou": 0.8},
    ]
    rows = {r["bucket"]: r for r in failure_buckets(details)}
    assert rows["centre_error_too_large"]["count"] == 1
    assert rows["box_size_error"]["count"] == 1
