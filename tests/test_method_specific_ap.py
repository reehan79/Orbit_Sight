"""AP must depend on the continuous score ranking, not only boxes."""

from __future__ import annotations

import numpy as np

from orbitsight.evaluation.tii_style import match_and_score


def test_different_score_rankings_change_ap_with_identical_boxes():
    # One GT; two preds — boxes identical across rankings; only scores swapped.
    gt = [(0, 40000, 10.0, 10.0, 4.0, 4.0)]
    # Pred A: high score on the matching box; Pred B: high score on the miss.
    match_box = (0, 40000, 10.0, 10.0, 4.0, 4.0, 0.9)
    miss_box = (0, 40000, 100.0, 100.0, 4.0, 4.0, 0.1)
    ranking_good = [match_box, miss_box]
    ranking_bad = [
        (0, 40000, 100.0, 100.0, 4.0, 4.0, 0.9),
        (0, 40000, 10.0, 10.0, 4.0, 4.0, 0.1),
    ]
    ap_good = match_and_score(gt, ranking_good).ap50
    ap_bad = match_and_score(gt, ranking_bad).ap50
    assert ap_good > ap_bad
    # Boxes present are the same set; only ranking differs.
    boxes_good = {(r[2], r[3], r[4], r[5]) for r in ranking_good}
    boxes_bad = {(r[2], r[3], r[4], r[5]) for r in ranking_bad}
    assert boxes_good == boxes_bad
