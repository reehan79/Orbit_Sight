from __future__ import annotations

import numpy as np

PATCH_SIZE = 32
PATCH_CHANNELS = 4
ROI_HALF_CELLS = 2.0


def rasterize_event_patch(
    current: np.ndarray,
    prior: np.ndarray,
    cx: float,
    cy: float,
    cell: float,
    start_us: int,
    end_us: int,
) -> np.ndarray:
    """Build a fixed 4x32x32 local event patch around a RAW candidate centre.

    Channels:
      CH0: log1p(current positive counts)
      CH1: log1p(current negative counts)
      CH2: mean normalized event time in [0,1] for occupied current bins (else 0)
      CH3: log1p(prior 80-ms all-polarity counts)
    """
    half = ROI_HALF_CELLS * float(cell)
    x0, y0 = cx - half, cy - half
    span = max(2.0 * half, 1e-6)
    duration = max(float(end_us - start_us), 1.0)

    pos_counts = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float64)
    neg_counts = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float64)
    time_sum = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float64)
    time_n = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float64)
    prior_counts = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float64)

    def _accumulate(events: np.ndarray, *, is_prior: bool) -> None:
        if len(events) == 0:
            return
        x = events[:, 0].astype(np.float64)
        y = events[:, 1].astype(np.float64)
        inside = (x >= x0) & (x <= x0 + span) & (y >= y0) & (y <= y0 + span)
        if not np.any(inside):
            return
        x = x[inside]
        y = y[inside]
        bx = np.clip(((x - x0) / span * PATCH_SIZE).astype(np.int64), 0, PATCH_SIZE - 1)
        by = np.clip(((y - y0) / span * PATCH_SIZE).astype(np.int64), 0, PATCH_SIZE - 1)
        if is_prior:
            np.add.at(prior_counts, (by, bx), 1.0)
            return
        pol = events[inside, 2].astype(np.int8) if events.shape[1] >= 3 else np.zeros(len(x), dtype=np.int8)
        pos = pol == 1
        neg = ~pos
        if np.any(pos):
            np.add.at(pos_counts, (by[pos], bx[pos]), 1.0)
        if np.any(neg):
            np.add.at(neg_counts, (by[neg], bx[neg]), 1.0)
        if events.shape[1] >= 4:
            t_norm = np.clip((events[inside, 3].astype(np.float64) - float(start_us)) / duration, 0.0, 1.0)
            np.add.at(time_sum, (by, bx), t_norm)
            np.add.at(time_n, (by, bx), 1.0)

    _accumulate(current, is_prior=False)
    _accumulate(prior, is_prior=True)

    recency = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float64)
    occupied = time_n > 0
    recency[occupied] = time_sum[occupied] / time_n[occupied]

    return np.stack(
        [np.log1p(pos_counts), np.log1p(neg_counts), recency, np.log1p(prior_counts)],
        axis=0,
    ).astype(np.float16)
