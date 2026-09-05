"""Parity tests for TRUE P1 fast path vs reference."""

from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from orbitsight.inference.b_current import SequenceStream
from orbitsight.inference.p1_detector import emit_tii_row, run_p1_window_fast, run_p1_window_reference
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows
from orbitsight.proposals import RawGridProposer


def _make_stream():
    rng = np.random.default_rng(2)
    w, h, c = 346, 260, 8
    n = 1200
    gx, gy = 14, 10
    xs = gx * c + rng.integers(0, c, size=n)
    ys = gy * c + rng.integers(0, c, size=n)
    pol = rng.integers(0, 2, size=n)
    t0 = 2_000_000
    ts = t0 + rng.integers(0, WINDOW_US * 8, size=n)
    order = np.argsort(ts)
    arr = np.stack([xs[order], ys[order], pol[order], ts[order]], axis=1).astype(np.float64)

    class Stream:
        sequence = "synthetic"
        width = w
        height = h
        cell = c
        timestamps = arr[:, 3]
        proposer = RawGridProposer(w, h, c, top_k=20)
        sensor = "DAVIS"

        def slice_window(self, start_us, end_us):
            left = int(np.searchsorted(self.timestamps, start_us, side="left"))
            right = int(np.searchsorted(self.timestamps, end_us, side="left"))
            prior_left = int(np.searchsorted(self.timestamps, start_us - 80_000, side="left"))
            current = np.asarray(arr[left:right, :4])
            prior = np.asarray(arr[prior_left:left, :4])
            return current, prior

    return Stream()


def _models():
    X = np.random.randn(80, 15).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int8)
    conf = ExtraTreesClassifier(
        n_estimators=16, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=1
    ).fit(X, y)
    Xs = np.random.randn(40, 33).astype(np.float32)
    ys = np.full((40, 2), math.log(2.0), dtype=np.float32)
    size = ExtraTreesRegressor(
        n_estimators=8, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=1
    ).fit(Xs, ys)
    return conf, size


def test_p1_reference_vs_fast_bit_identical():
    stream = _make_stream()
    conf, size = _models()
    threshold = 0.3
    ref_rows = []
    fast_rows = []
    for ws in enumerate_challenge_windows(stream.timestamps)[:40]:
        we = int(ws) + WINDOW_US
        ref = run_p1_window_reference(stream, int(ws), we, conf, size, threshold=threshold)
        fast = run_p1_window_fast(stream, int(ws), we, conf, size, threshold=threshold)
        if ref is None and fast is None:
            continue
        assert ref is not None and fast is not None
        assert ref.emitted == fast.emitted
        if ref.emitted:
            ref_rows.append(emit_tii_row(ref))
            fast_rows.append(emit_tii_row(fast))
            assert abs(ref.confidence - fast.confidence) <= 1e-12
            for a, b in zip(emit_tii_row(ref), emit_tii_row(fast)):
                if isinstance(a, float):
                    assert abs(a - b) <= 1e-12
                else:
                    assert a == b
    assert ref_rows == fast_rows
