from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def aggregate_detection_metrics(details: list[dict]) -> dict[str, float]:
    """Aggregate GT-level detection records into pooled micro, sequence macro, and fold mean."""
    if not details:
        return {"n_gt": 0.0}

    ious = [float(d["iou"]) for d in details]
    iou50 = [float(d.get("iou50", float(d["iou"] >= 0.5))) for d in details]
    iou75 = [float(d.get("iou75", float(d["iou"] >= 0.75))) for d in details]
    n_gt = len(details)

    by_seq: dict[str, list[float]] = defaultdict(list)
    by_fold: dict[int, list[float]] = defaultdict(list)
    for d, hit50 in zip(details, iou50, strict=True):
        by_seq[str(d["sequence"])].append(hit50)
        by_fold[int(d["fold"])].append(hit50)

    centre = [
        float(d["centre_error"])
        for d in details
        if "centre_error" in d and not math.isnan(float(d["centre_error"]))
    ]

    ious_arr = np.asarray(ious, dtype=np.float64)
    centre_arr = np.asarray(centre, dtype=np.float64) if centre else np.empty(0)

    out: dict[str, float] = {
        "n_gt": float(n_gt),
        "pooled_micro_iou50_pct": 100.0 * sum(iou50) / n_gt,
        "pooled_micro_iou75_pct": 100.0 * sum(iou75) / n_gt,
        "sequence_macro_iou50_pct": 100.0
        * sum(sum(v) / len(v) for v in by_seq.values())
        / len(by_seq),
        "fold_mean_iou50_pct": 100.0
        * sum(sum(v) / len(v) for v in by_fold.values())
        / len(by_fold),
        "mean_iou": float(np.mean(ious_arr)),
        "median_iou": float(np.median(ious_arr)),
    }
    if len(centre_arr):
        out["centre_error_mean"] = float(np.mean(centre_arr))
        out["centre_error_median"] = float(np.median(centre_arr))
        out["centre_error_p90"] = float(np.percentile(centre_arr, 90))
    return out


def failure_buckets(details: list[dict]) -> list[dict]:
    """Classify GT records into success and failure buckets with dual percentage columns."""
    buckets: dict[str, int] = defaultdict(int)
    for row in details:
        iou = float(row["iou"])
        if iou >= 0.5:
            buckets["success_iou50"] += 1
        elif not row["proposal_hit"]:
            buckets["proposal_miss"] += 1
        elif not row["ranker_hit"]:
            buckets["ranking_error"] += 1
        elif float(row["oracle_iou"]) < 0.5:
            buckets["centre_error_too_large"] += 1
        elif iou < 0.5:
            buckets["box_size_error"] += 1
        else:
            buckets["unclassified_failure"] += 1

    total_gt = max(sum(buckets.values()), 1)
    total_failures = max(total_gt - buckets.get("success_iou50", 0), 1)
    rows: list[dict] = []
    for bucket, count in sorted(buckets.items()):
        rows.append(
            {
                "bucket": bucket,
                "count": count,
                "pct_all_gt": 100.0 * count / total_gt,
                "pct_failures_only": 100.0 * count / total_failures if bucket != "success_iou50" else 0.0,
            }
        )
    return rows
