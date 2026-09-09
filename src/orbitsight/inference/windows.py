from __future__ import annotations

import numpy as np

WINDOW_US = 40_000


def enumerate_challenge_windows(timestamps: np.ndarray) -> np.ndarray:
    """Enumerate half-open 40-ms windows matching TII DataLoader convention.

    From OrbitSight_DataLoader/visualize_dataset.py:
      t0 = int(ev_t[0]); t1 = int(ev_t[-1])
      window_starts = np.arange(t0, t1, WINDOW_US)
      window i covers [ws, ws + WINDOW_US)

    Returns starts as int64 array.
    """
    if timestamps is None or len(timestamps) == 0:
        return np.empty(0, dtype=np.int64)
    t0 = int(timestamps[0])
    t1 = int(timestamps[-1])
    if t1 <= t0:
        return np.empty(0, dtype=np.int64)
    return np.arange(t0, t1, WINDOW_US, dtype=np.int64)


def window_end(start_us: int) -> int:
    return int(start_us) + WINDOW_US
