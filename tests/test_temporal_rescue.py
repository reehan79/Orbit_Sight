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
    no_rescue_threshold,
    select_rescue_threshold_from_f1,
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
    feats = compute_temporal_features(
        hist,
        cx_cells=10.0,
        cy_cells=10.0,
        gate_prob=0.3,
        base_conf=0.4,
        event_rate_log=2.0,
        event_count=100.0,
    )
    assert feats[2] == 2.0
    expected_w = DECAY**2 + DECAY**3
    assert abs(feats[3] - expected_w) < 1e-9


def test_motion_support_prefers_matching_velocity():
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
    assert feats[4] > 0
    assert feats[6] == 1.0
    assert feats[7] == 0.0
    assert feats[5] == 2.0


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


def test_empty_windows_advance_history_lag():
    """Candidate at t0, six empties, candidate at t7 => t0 is lag 7 not lag 1."""
    state = CausalTemporalState(HISTORY)
    state.push(HistorySlot(10.0, 10.0, 0.9, 0.8, 100.0, has_candidate=True))
    for _ in range(6):
        state.push(HistorySlot.empty(event_count=5.0))
    assert len(state) == 7
    feats = state.features(
        cx_cells=10.0,
        cy_cells=10.0,
        gate_prob=0.5,
        base_conf=0.5,
        event_rate_log=1.0,
        event_count=10.0,
    )
    # Only lag-7 slot has a candidate near current centre
    assert feats[2] == 1.0  # persistence_count_7
    assert abs(feats[3] - (DECAY**7)) < 1e-12
    # If empties were skipped, lag would be 1 and weight would be DECAY**1
    assert abs(feats[3] - DECAY) > 0.1


def test_no_rescue_threshold_selected_when_all_rescues_hurt():
    scores = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
    no_rescue = no_rescue_threshold(scores)
    assert no_rescue > float(scores.max())
    # Every emitting threshold lowers F1 vs D2-only baseline
    baseline_f1 = 0.80
    f1_at = {no_rescue: baseline_f1}
    for t in (0.1, 0.5, 0.9):
        f1_at[t] = baseline_f1 - 0.05
    chosen = select_rescue_threshold_from_f1(
        [0.1, 0.5, 0.9, no_rescue], f1_at, no_rescue
    )
    assert chosen == no_rescue
