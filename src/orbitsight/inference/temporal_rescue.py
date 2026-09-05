"""Causal temporal features for selective D2 rejection rescue.

Does not alter the accepted D2 path. History is previous windows only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

HISTORY = 7
DECAY = 0.90
VEL_MIN = -3
VEL_MAX = 3
MATCH_CELLS = 1.5
NO_SUPPORT_RESIDUAL = 10.0
EPS = 1e-6

TEMPORAL_FEATURE_NAMES = (
    "current_d2_gate_probability",
    "current_base_candidate_confidence",
    "persistence_count_7",
    "persistence_weighted",
    "best_motion_support",
    "best_motion_hits",
    "best_velocity_x",
    "best_velocity_y",
    "best_motion_residual",
    "prior_gate_max",
    "prior_gate_mean",
    "prior_base_conf_max",
    "prior_base_conf_mean",
    "current_event_rate_log",
    "current_to_previous_event_rate_ratio",
)

N_TEMPORAL_FEATURES = len(TEMPORAL_FEATURE_NAMES)


@dataclass(frozen=True)
class HistorySlot:
    """Compact causal state for one previous window."""

    cx_cells: float
    cy_cells: float
    gate_prob: float
    base_conf: float
    event_count: float


class CausalTemporalState:
    """Incremental previous-window history (no full disk rescan)."""

    def __init__(self, horizon: int = HISTORY) -> None:
        self.horizon = int(horizon)
        self._hist: deque[HistorySlot] = deque(maxlen=self.horizon)

    def clear(self) -> None:
        self._hist.clear()

    def __len__(self) -> int:
        return len(self._hist)

    def push(self, slot: HistorySlot) -> None:
        self._hist.append(slot)

    def features(
        self,
        *,
        cx_cells: float,
        cy_cells: float,
        gate_prob: float,
        base_conf: float,
        event_rate_log: float,
        event_count: float,
    ) -> np.ndarray:
        return compute_temporal_features(
            list(self._hist),
            cx_cells=cx_cells,
            cy_cells=cy_cells,
            gate_prob=gate_prob,
            base_conf=base_conf,
            event_rate_log=event_rate_log,
            event_count=event_count,
        )


def compute_temporal_features(
    history: list[HistorySlot],
    *,
    cx_cells: float,
    cy_cells: float,
    gate_prob: float,
    base_conf: float,
    event_rate_log: float,
    event_count: float,
) -> np.ndarray:
    """Compute the fixed 15 temporal features from causal history only."""
    hist = history[-HISTORY:] if history else []
    n = len(hist)

    persistence_count = 0
    persistence_weighted = 0.0
    for lag in range(1, n + 1):
        slot = hist[-lag]
        dist = math_hypot(cx_cells - slot.cx_cells, cy_cells - slot.cy_cells)
        if dist <= MATCH_CELLS:
            persistence_count += 1
            persistence_weighted += DECAY**lag

    best_support = 0.0
    best_hits = 0
    best_vx = 0
    best_vy = 0
    best_residual = NO_SUPPORT_RESIDUAL
    if n > 0:
        for vx in range(VEL_MIN, VEL_MAX + 1):
            for vy in range(VEL_MIN, VEL_MAX + 1):
                support = 0.0
                hits = 0
                w_sum = 0.0
                d_sum = 0.0
                for lag in range(1, n + 1):
                    slot = hist[-lag]
                    pred_x = cx_cells - lag * vx
                    pred_y = cy_cells - lag * vy
                    dist = math_hypot(slot.cx_cells - pred_x, slot.cy_cells - pred_y)
                    if dist <= MATCH_CELLS:
                        w = DECAY**lag
                        support += w
                        hits += 1
                        w_sum += w
                        d_sum += w * dist
                if support > best_support or (
                    support == best_support and hits > best_hits
                ):
                    best_support = support
                    best_hits = hits
                    best_vx = vx
                    best_vy = vy
                    best_residual = (d_sum / w_sum) if w_sum > 0 else NO_SUPPORT_RESIDUAL

    if n == 0:
        prior_gate_max = 0.0
        prior_gate_mean = 0.0
        prior_base_max = 0.0
        prior_base_mean = 0.0
        rate_ratio = 1.0
    else:
        gates = [s.gate_prob for s in hist]
        confs = [s.base_conf for s in hist]
        prior_gate_max = float(max(gates))
        prior_gate_mean = float(sum(gates) / n)
        prior_base_max = float(max(confs))
        prior_base_mean = float(sum(confs) / n)
        mean_prev_count = float(sum(s.event_count for s in hist) / n)
        rate_ratio = float(event_count) / (mean_prev_count + EPS)

    out = np.asarray(
        [
            float(gate_prob),
            float(base_conf),
            float(persistence_count),
            float(persistence_weighted),
            float(best_support),
            float(best_hits),
            float(best_vx),
            float(best_vy),
            float(best_residual),
            prior_gate_max,
            prior_gate_mean,
            prior_base_max,
            prior_base_mean,
            float(event_rate_log),
            rate_ratio,
        ],
        dtype=np.float64,
    )
    assert out.shape == (N_TEMPORAL_FEATURES,)
    return out


def math_hypot(dx: float, dy: float) -> float:
    return float(np.hypot(dx, dy))
