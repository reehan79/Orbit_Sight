from __future__ import annotations

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
