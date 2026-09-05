"""Parity: gate features with reused geometry match recomputed features."""

from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from orbitsight.inference.p1_detector import build_gate_features, run_p1_window_fast
from orbitsight.inference.windows import WINDOW_US
from orbitsight.proposals import RawGridProposer


def test_gate_feature_reuse_matches_recompute():
    rng = np.random.default_rng(7)
    w, h, c = 346, 260, 8
    n = 900
    gx, gy = 11, 9
    xs = gx * c + rng.integers(0, c, size=n)
    ys = gy * c + rng.integers(0, c, size=n)
    pol = rng.integers(0, 2, size=n)
    t0 = 3_000_000
    ts = t0 + rng.integers(0, WINDOW_US, size=n)
    order = np.argsort(ts)
    arr = np.stack([xs[order], ys[order], pol[order], ts[order]], axis=1).astype(np.float64)

    class Stream:
        width = w
        height = h
        cell = c
        timestamps = arr[:, 3]
        proposer = RawGridProposer(w, h, c, top_k=20)

        def slice_window(self, start_us, end_us):
            left = int(np.searchsorted(self.timestamps, start_us, side="left"))
            right = int(np.searchsorted(self.timestamps, end_us, side="left"))
            prior_left = int(np.searchsorted(self.timestamps, start_us - 80_000, side="left"))
            return np.asarray(arr[left:right, :4]), np.asarray(arr[prior_left:left, :4])

    X = rng.standard_normal((60, 15)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int8)
    conf = ExtraTreesClassifier(n_estimators=12, max_depth=5, random_state=42, n_jobs=1).fit(X, y)
    Xs = rng.standard_normal((40, 33)).astype(np.float32)
    ys = np.full((40, 2), math.log(2.0), dtype=np.float32)
    size = ExtraTreesRegressor(n_estimators=8, max_depth=4, random_state=42, n_jobs=1).fit(Xs, ys)

    stream = Stream()
    res = run_p1_window_fast(stream, t0, t0 + WINDOW_US, conf, size, always_emit=True)
    assert res is not None and res.geometry is not None
    a = build_gate_features(res, stream, size, reuse_geometry=True)
    b = build_gate_features(res, stream, size, reuse_geometry=False)
    assert np.allclose(a, b, atol=1e-12, rtol=0)
