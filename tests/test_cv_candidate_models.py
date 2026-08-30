import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cv_candidate_models import ranking_metrics


def test_mrr_includes_groups_without_positive_candidate():
    data = {
        "sequence": np.asarray(["seq_a", "seq_a"], dtype=object),
        "target": np.asarray([0, 1], dtype=np.int8),
    }
    groups = [np.asarray([0, 1], dtype=np.int64)]
    scores = np.asarray([0.9, 0.1], dtype=np.float32)
    lookup = {0: 0, 1: 1}

    metrics = ranking_metrics(data, groups, scores, lookup)
    assert metrics["mrr"] == 0.5
    assert metrics["top1_micro"] == 0.0
    assert metrics["top3_micro"] == 1.0


def test_mrr_two_groups_one_positive_at_rank_two_one_without_positive():
    data = {
        "sequence": np.asarray(["seq_a", "seq_a", "seq_b"], dtype=object),
        "target": np.asarray([0, 1, 0], dtype=np.int8),
    }
    groups = [
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2], dtype=np.int64),
    ]
    scores = np.asarray([0.9, 0.1, 0.5], dtype=np.float32)
    lookup = {0: 0, 1: 1, 2: 2}

    metrics = ranking_metrics(data, groups, scores, lookup)
    assert metrics["mrr"] == 0.25
