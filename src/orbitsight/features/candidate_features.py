from __future__ import annotations

import numpy as np

from orbitsight.proposals import Candidate

FEATURE_NAMES = (
    "candidate_score",
    "log_candidate_count",
    "rank_fraction",
    "cx_normalized",
    "cy_normalized",
    "log_local_count",
    "local_event_fraction",
    "local_positive_fraction",
    "local_unique_pixel_fraction",
    "local_spread_x_cells",
    "local_spread_y_cells",
    "log_prior_local_count",
    "prior_to_current_rate_ratio",
    "log_global_event_count",
    "global_positive_fraction",
)


def _valid_xyz(events: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(events) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int8),
        )
    x = events[:, 0].astype(np.int64, copy=False)
    y = events[:, 1].astype(np.int64, copy=False)
    if events.shape[1] >= 3:
        p = events[:, 2].astype(np.int8, copy=False)
    else:
        p = np.zeros(len(events), dtype=np.int8)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid], p[valid]


def extract_candidate_features(
    current_events: np.ndarray,
    prior_events: np.ndarray,
    candidates: list[Candidate],
    width: int,
    height: int,
    cell_size: int,
) -> np.ndarray:
    """Build label-free numerical features for candidate ranking.

    The function intentionally reads only x/y/polarity from the event stream.  Event
    labels in column 4 are never used as inputs.  `prior_events` is expected to cover
    approximately 80 ms when `current_events` covers a 40-ms challenge window.
    """
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    x, y, p = _valid_xyz(current_events, width, height)
    px, py, _ = _valid_xyz(prior_events, width, height)
    global_count = max(len(x), 1)
    global_positive = float(np.mean(p == 1)) if len(p) else 0.0
    radius = float(cell_size) * 1.5

    rows: list[list[float]] = []
    denom_rank = max(len(candidates) - 1, 1)

    for rank0, candidate in enumerate(candidates):
        local_mask = (np.abs(x - candidate.cx) <= radius) & (np.abs(y - candidate.cy) <= radius)
        lx = x[local_mask]
        ly = y[local_mask]
        lp = p[local_mask]
        local_count = len(lx)

        prior_mask = (np.abs(px - candidate.cx) <= radius) & (np.abs(py - candidate.cy) <= radius)
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

        # Prior interval is nominally twice as long as the current 40-ms interval.
        prior_rate = prior_count / 2.0
        current_rate = max(local_count, 1)

        rows.append(
            [
                float(candidate.score),
                float(np.log1p(candidate.count)),
                float(rank0 / denom_rank),
                float(candidate.cx / max(width, 1)),
                float(candidate.cy / max(height, 1)),
                float(np.log1p(local_count)),
                float(local_count / global_count),
                local_positive,
                unique_fraction,
                spread_x,
                spread_y,
                float(np.log1p(prior_count)),
                float(prior_rate / current_rate),
                float(np.log1p(len(x))),
                global_positive,
            ]
        )

    return np.asarray(rows, dtype=np.float32)
