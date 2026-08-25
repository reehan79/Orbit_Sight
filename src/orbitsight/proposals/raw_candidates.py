from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

SENSOR_GEOMETRY = {"DAVIS": (346, 260, 8), "DVX": (640, 480, 12), "EVK4": (1280, 720, 16)}


@dataclass(frozen=True)
class Candidate:
    cx: float
    cy: float
    score: float
    count: int
    grid_x: int
    grid_y: int


def infer_sensor_geometry(sequence: str) -> tuple[int, int, int]:
    upper = sequence.upper()
    if upper.startswith("DAVIS"):
        return SENSOR_GEOMETRY["DAVIS"]
    if upper.startswith("DVX"):
        return SENSOR_GEOMETRY["DVX"]
    if "EVK4" in upper:
        return SENSOR_GEOMETRY["EVK4"]
    raise ValueError(f"Unknown sensor for sequence {sequence!r}")


class RawGridProposer:
    """Label-free high-recall proposal baseline with no hard hot-pixel deletion."""
    def __init__(self, width: int, height: int, cell_size: int, top_k: int = 20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.top_k = top_k
        self.grid_w = math.ceil(width / cell_size)
        self.grid_h = math.ceil(height / cell_size)

    def propose(self, events: np.ndarray) -> list[Candidate]:
        if events.ndim != 2 or events.shape[1] < 2:
            raise ValueError("events must have shape (N, >=2) with x,y in columns 0,1")
        if len(events) == 0:
            return []
        x = events[:, 0].astype(np.int64, copy=False)
        y = events[:, 1].astype(np.int64, copy=False)
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y = x[valid], y[valid]
        if len(x) == 0:
            return []
        gx, gy = x // self.cell_size, y // self.cell_size
        ids = gy * self.grid_w + gx
        counts = np.bincount(ids, minlength=self.grid_w * self.grid_h)
        occupied = np.flatnonzero(counts)
        if len(occupied) == 0:
            return []
        k = min(self.top_k, len(occupied))
        if len(occupied) > k:
            local = np.argpartition(counts[occupied], -k)[-k:]
            ranked = occupied[local]
        else:
            ranked = occupied
        ranked = ranked[np.argsort(counts[ranked])[::-1]]
        max_count = max(int(counts[ranked[0]]), 1)
        out: list[Candidate] = []
        for gid in ranked:
            gy_i, gx_i = divmod(int(gid), self.grid_w)
            count = int(counts[gid])
            out.append(Candidate(gx_i * self.cell_size + self.cell_size / 2.0, gy_i * self.cell_size + self.cell_size / 2.0, count / max_count, count, gx_i, gy_i))
        return out
