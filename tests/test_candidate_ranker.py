import numpy as np
import pytest

pytest.importorskip("sklearn")

from orbitsight.models import fit_bbox_ridge, fit_rankers, score_ranker


def test_minimal_rankers_fit_and_score():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 15)).astype(np.float32)
    y = np.zeros(80, dtype=np.int8)
    y[::5] = 1
    X[y == 1, 0] += 2.0
    ranks = np.tile(np.arange(1, 21), 4)

    models = fit_rankers(X, y, ranks)
    assert set(models) == {"M0_raw_rank", "M1_logistic", "M2_hist_gb"}
    for bundle in models.values():
        scores = score_ranker(bundle, X, ranks)
        assert scores.shape == (80,)
        assert np.all(np.isfinite(scores))


def test_bbox_ridge_returns_four_values():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(50, 15)).astype(np.float32)
    targets = rng.normal(size=(50, 4)).astype(np.float32)
    model = fit_bbox_ridge(X, targets)
    pred = model.predict(X[:3])
    assert pred.shape == (3, 4)
    assert np.all(np.isfinite(pred))
