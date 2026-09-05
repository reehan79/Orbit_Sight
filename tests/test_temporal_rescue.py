"""Unit tests for causal temporal rescue features (no dataset I/O)."""

from __future__ import annotations

import numpy as np

from orbitsight.inference.temporal_rescue import (
    DECAY,
    HISTORY,
    MATCH_CELLS,
    N_TEMPORAL_FEATURES,
    NO_SUPPORT_RESIDUAL,
    TEMPORAL_FEATURE_NAMES,
    CausalTemporalState,
    HistorySlot,
    compute_temporal_features,
)


def test_feature_dim_and_names():
    assert len(TEMPORAL_FEATURE_NAMES) == 15
    assert N_TEMPORAL_FEATURES == 15


def test_empty_history_defaults():
    feats = compute_temporal_features(
        [],
        cx_cells=10.0,
        cy_cells=10.0,
        gate_prob=0.2,
        base_conf=0.4,
        event_rate_log=1.5,
        event_count=10.0,
    )
    assert feats.shape == (15,)
    assert feats[0] == 0.2
    assert feats[1] == 0.4
    assert feats[2] == 0.0  # persistence_count
    assert feats[4] == 0.0  # best_motion_support
    assert feats[8] == NO_SUPPORT_RESIDUAL
    assert feats[13] == 1.5


def test_persistence_zero_motion():
    hist = [
        HistorySlot(10.0, 10.0, 0.5, 0.5, 100.0),
        HistorySlot(10.2, 10.1, 0.6, 0.55, 110.0),
        HistorySlot(20.0, 20.0, 0.1, 0.1, 50.0),  # far
    ]
    # Current near first two (lags from end): lag1=far, lag2=near, lag3=near
    state_hist = hist  # chronological oldest→newest
    feats = compute_temporal_features(
        state_hist,
        cx_cells=10.0,
        cy_cells=10.0,
        gate_prob=0.3,
        base_conf=0.4,
        event_rate_log=2.0,
        event_count=100.0,
    )
    # lag1 = (20,20) far; lag2=(10.2,10.1) near; lag3=(10,10) near
    assert feats[2] == 2.0
    expected_w = DECAY**2 + DECAY**3
    assert abs(feats[3] - expected_w) < 1e-9


def test_motion_support_prefers_matching_velocity():
    # Moving +1,+0 cells per window; history at t-1=(9,10), t-2=(8,10)
    hist = [
        HistorySlot(8.0, 10.0, 0.5, 0.5, 80.0),
        HistorySlot(9.0, 10.0, 0.5, 0.5, 90.0),
    ]
    feats = compute_temporal_features(
        hist,
        cx_cells=10.0,
        cy_cells=10.0,
        gate_prob=0.4,
        base_conf=0.5,
        event_rate_log=1.0,
        event_count=100.0,
    )
    assert feats[4] > 0  # best_motion_support
    assert feats[6] == 1.0  # best_velocity_x
    assert feats[7] == 0.0  # best_velocity_y
    assert feats[5] == 2.0  # hits


def test_causal_state_incremental_no_future():
    state = CausalTemporalState(HISTORY)
    f0 = state.features(
        cx_cells=1.0,
        cy_cells=1.0,
        gate_prob=0.1,
        base_conf=0.2,
        event_rate_log=0.0,
        event_count=1.0,
    )
    assert f0[2] == 0.0
    state.push(HistorySlot(1.0, 1.0, 0.1, 0.2, 1.0))
    f1 = state.features(
        cx_cells=1.0,
        cy_cells=1.0,
        gate_prob=0.2,
        base_conf=0.3,
        event_rate_log=0.0,
        event_count=2.0,
    )
    assert f1[2] == 1.0
    assert abs(f1[3] - DECAY) < 1e-9


def test_match_radius_constant():
    assert MATCH_CELLS == 1.5
    assert HISTORY == 7
