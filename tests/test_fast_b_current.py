from __future__ import annotations

import numpy as np

from orbitsight.inference.b_current import SequenceStream, run_b_current_fast, run_b_current_reference
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows
from orbitsight.proposals import Candidate, RawGridProposer
from orbitsight.features import extract_candidate_features
from orbitsight.models.candidate_ranker import RankerBundle


def test_enumerate_matches_tii_convention():
    # t0=1000000, t1=1000000+120000 -> starts at 1e6, 1e6+40k, 1e6+80k
    ts = np.array([1_000_000, 1_020_000, 1_120_000], dtype=np.int64)
    starts = enumerate_challenge_windows(ts)
    assert list(starts) == [1_000_000, 1_040_000, 1_080_000]


def test_fast_features_match_reference():
    from orbitsight.features.candidate_features import extract_candidate_features
    from orbitsight.features.candidate_features_fast import extract_candidate_features_fast

    rng = np.random.default_rng(1)
    width, height, cell = 346, 260, 8
    n = 500
    current = np.column_stack(
        [
            rng.integers(0, width, n),
            rng.integers(0, height, n),
            rng.integers(0, 2, n),
            rng.integers(0, 40_000, n),
        ]
    ).astype(np.float64)
    prior = current.copy()
    proposer = RawGridProposer(width, height, cell, top_k=20)
    cands = proposer.propose(current)
    a = extract_candidate_features(current, prior, cands, width, height, cell)
    b = extract_candidate_features_fast(current, prior, cands, width, height, cell)
    assert np.allclose(a, b, atol=1e-6, rtol=0)


class _DummyRanker:
    def predict_proba(self, X):
        # Prefer earlier candidates slightly via feature[0]
        p1 = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.stack([1 - p1, p1], axis=1)


def test_fast_b_current_parity_synthetic():
    import math

    rng = np.random.default_rng(0)
    width, height, cell = 346, 260, 8
    n = 800
    gx, gy = 10, 12
    xs = gx * cell + rng.integers(0, cell, size=n)
    ys = gy * cell + rng.integers(0, cell, size=n)
    pol = rng.integers(0, 2, size=n)
    t0 = 1_000_000
    ts = t0 + rng.integers(0, WINDOW_US, size=n)
    order = np.argsort(ts)
    current = np.stack([xs[order], ys[order], pol[order], ts[order]], axis=1).astype(np.float64)
    prior = current.copy()
    prior[:, 3] -= 80_000

    proposer = RawGridProposer(width, height, cell, top_k=20)

    class S:
        pass

    stream = S()
    stream.width = width
    stream.height = height
    stream.cell = cell
    stream.proposer = proposer
    stream.slice_window = lambda a, b: (current, prior)

    from sklearn.ensemble import ExtraTreesRegressor

    X = np.random.randn(50, 33).astype(np.float32)
    y = np.full((50, 2), math.log(2.0), dtype=np.float32)
    size = ExtraTreesRegressor(
        n_estimators=8, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=1
    ).fit(X, y)

    bundle = RankerBundle("M2b_extra_trees", _DummyRanker())
    ref = run_b_current_reference(stream, t0, t0 + WINDOW_US, bundle, size)
    fast = run_b_current_fast(stream, t0, t0 + WINDOW_US, bundle, size)
    assert ref.selected is not None and fast.selected is not None
    assert (ref.selected.grid_x, ref.selected.grid_y) == (fast.selected.grid_x, fast.selected.grid_y)
    assert np.allclose(ref.features, fast.features, atol=1e-6)
    assert np.allclose(ref.scores, fast.scores, atol=1e-6)
    assert abs(ref.cx - fast.cx) <= 1e-6
    assert abs(ref.cy - fast.cy) <= 1e-6
    assert abs(ref.width - fast.width) <= 1e-5
    assert abs(ref.height - fast.height) <= 1e-5
