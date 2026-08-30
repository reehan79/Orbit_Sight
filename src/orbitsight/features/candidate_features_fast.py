from __future__ import annotations

import numpy as np

from orbitsight.features.candidate_features import FEATURE_NAMES, _valid_xyz
from orbitsight.proposals import Candidate


def extract_candidate_features_fast(
    current_events: np.ndarray,
    prior_events: np.ndarray,
    candidates: list[Candidate],
    width: int,
    height: int,
    cell_size: int,
) -> np.ndarray:
    """Same definitions as extract_candidate_features; fewer Python/array round-trips."""
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    x, y, p = _valid_xyz(current_events, width, height)
    px, py, _ = _valid_xyz(prior_events, width, height)
    # Contiguous float64 views for distance math (matches original promotion behaviour).
    xf = np.asarray(x, dtype=np.float64)
    yf = np.asarray(y, dtype=np.float64)
    pxf = np.asarray(px, dtype=np.float64)
    pyf = np.asarray(py, dtype=np.float64)

    global_count = max(len(x), 1)
    global_positive = float(np.mean(p == 1)) if len(p) else 0.0
    radius = float(cell_size) * 1.5
    denom_rank = max(len(candidates) - 1, 1)
    log_global = float(np.log1p(len(x)))

    k = len(candidates)
    out = np.empty((k, len(FEATURE_NAMES)), dtype=np.float32)
    for rank0, candidate in enumerate(candidates):
        local_mask = (np.abs(xf - candidate.cx) <= radius) & (np.abs(yf - candidate.cy) <= radius)
        lx = x[local_mask]
        ly = y[local_mask]
        lp = p[local_mask]
        local_count = len(lx)

        prior_mask = (np.abs(pxf - candidate.cx) <= radius) & (np.abs(pyf - candidate.cy) <= radius)
        prior_count = int(np.count_nonzero(prior_mask))

        if local_count:
            local_positive = float(np.mean(lp == 1))
            pixel_ids = ly * width + lx
            unique_fraction = float(len(np.unique(pixel_ids)) / local_count)
            spread_x = float(np.std(lx) / max(cell_size, 1))
            spread_y = float(np.std(ly) / max(cell_size, 1))
        else:
            local_positive = 0.0
            unique_fraction = 0.0
            spread_x = 0.0
            spread_y = 0.0

        prior_rate = prior_count / 2.0
        current_rate = max(local_count, 1)
        out[rank0, 0] = float(candidate.score)
        out[rank0, 1] = float(np.log1p(candidate.count))
        out[rank0, 2] = float(rank0 / denom_rank)
        out[rank0, 3] = float(candidate.cx / max(width, 1))
        out[rank0, 4] = float(candidate.cy / max(height, 1))
        out[rank0, 5] = float(np.log1p(local_count))
        out[rank0, 6] = float(local_count / global_count)
        out[rank0, 7] = local_positive
        out[rank0, 8] = unique_fraction
        out[rank0, 9] = spread_x
        out[rank0, 10] = spread_y
        out[rank0, 11] = float(np.log1p(prior_count))
        out[rank0, 12] = float(prior_rate / current_rate)
        out[rank0, 13] = log_global
        out[rank0, 14] = global_positive
    return out
