from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

LOCAL_GEOMETRY_NAMES = (
    "local_centroid_dx",
    "local_centroid_dy",
    "local_pos_centroid_dx",
    "local_pos_centroid_dy",
    "local_neg_centroid_dx",
    "local_neg_centroid_dy",
    "local_x_p10",
    "local_x_p50",
    "local_x_p90",
    "local_y_p10",
    "local_y_p50",
    "local_y_p90",
    "local_x_std",
    "local_y_std",
    "local_cov_xy",
    "local_event_count",
    "local_unique_pixels",
    "local_positive_fraction",
)


def roi_mask(events: np.ndarray, cx: float, cy: float, cell: float) -> np.ndarray:
    half = 2.0 * cell
    x = events[:, 0]
    y = events[:, 1]
    return (x >= cx - half) & (x <= cx + half) & (y >= cy - half) & (y <= cy + half)


def refine_c1_centroid(events: np.ndarray, cx: float, cy: float, cell: float) -> tuple[float, float]:
    mask = roi_mask(events, cx, cy, cell)
    roi = events[mask]
    if len(roi) == 0:
        return cx, cy
    return float(np.mean(roi[:, 0])), float(np.mean(roi[:, 1]))


def refine_c4_median(events: np.ndarray, cx: float, cy: float, cell: float) -> tuple[float, float]:
    mask = roi_mask(events, cx, cy, cell)
    roi = events[mask]
    if len(roi) == 0:
        return cx, cy
    return float(np.median(roi[:, 0])), float(np.median(roi[:, 1]))


def refine_c5_soft_background_centroid(
    current: np.ndarray,
    prior: np.ndarray,
    cx: float,
    cy: float,
    cell: float,
) -> tuple[float, float]:
    mask = roi_mask(current, cx, cy, cell)
    roi = current[mask]
    if len(roi) == 0:
        return cx, cy
    prior_counts: dict[tuple[int, int], int] = defaultdict(int)
    if len(prior) > 0:
        px = prior[:, 0].astype(np.int64)
        py = prior[:, 1].astype(np.int64)
        for x_val, y_val in zip(px, py, strict=True):
            prior_counts[(int(x_val), int(y_val))] += 1
    weights = np.empty(len(roi), dtype=np.float64)
    for i, event in enumerate(roi):
        key = (int(event[0]), int(event[1]))
        prior_n = prior_counts.get(key, 0)
        weights[i] = 1.0 / math.sqrt(1.0 + prior_n)
    wsum = float(weights.sum())
    if wsum <= 0.0:
        return cx, cy
    return float(np.sum(roi[:, 0] * weights) / wsum), float(np.sum(roi[:, 1] * weights) / wsum)


def local_extent_from_roi(events: np.ndarray, cx: float, cy: float, cell: float) -> tuple[float, float]:
    mask = roi_mask(events, cx, cy, cell)
    roi = events[mask]
    if len(roi) == 0:
        return 1.0, 1.0
    x = roi[:, 0].astype(np.float64)
    y = roi[:, 1].astype(np.float64)
    width = max(float(np.percentile(x, 90) - np.percentile(x, 10)), 1.0)
    height = max(float(np.percentile(y, 90) - np.percentile(y, 10)), 1.0)
    return width, height


def extract_local_geometry_features(
    events: np.ndarray,
    cx: float,
    cy: float,
    cell: float,
    width: int,
    height: int,
) -> np.ndarray:
    mask = roi_mask(events, cx, cy, cell)
    roi = events[mask]
    if len(roi) == 0:
        return np.zeros(len(LOCAL_GEOMETRY_NAMES), dtype=np.float32)

    x = roi[:, 0].astype(np.float64)
    y = roi[:, 1].astype(np.float64)
    if roi.shape[1] >= 3:
        p = roi[:, 2].astype(np.int8)
    else:
        p = np.zeros(len(roi), dtype=np.int8)

    rdx = x - cx
    rdy = y - cy
    pos = p == 1
    neg = p == 0
    out = [
        float(np.mean(rdx)),
        float(np.mean(rdy)),
        float(np.mean(rdx[pos])) if np.any(pos) else 0.0,
        float(np.mean(rdy[pos])) if np.any(pos) else 0.0,
        float(np.mean(rdx[neg])) if np.any(neg) else 0.0,
        float(np.mean(rdy[neg])) if np.any(neg) else 0.0,
        float(np.percentile(rdx, 10)),
        float(np.percentile(rdx, 50)),
        float(np.percentile(rdx, 90)),
        float(np.percentile(rdy, 10)),
        float(np.percentile(rdy, 50)),
        float(np.percentile(rdy, 90)),
        float(np.std(rdx)),
        float(np.std(rdy)),
        float(np.cov(rdx, rdy)[0, 1]) if len(rdx) > 1 else 0.0,
        float(len(roi)),
        float(len(np.unique((y.astype(np.int64) * width + x.astype(np.int64))))),
        float(np.mean(p == 1)),
    ]
    return np.asarray(out, dtype=np.float32)
