from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orbitsight.features import FEATURE_NAMES

FEATURE_COLUMNS = tuple(FEATURE_NAMES)


@dataclass
class RankerBundle:
    name: str
    model: Any | None


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int8)
    n = len(y)
    positives = max(int(np.count_nonzero(y == 1)), 1)
    negatives = max(int(np.count_nonzero(y == 0)), 1)
    weights = np.empty(n, dtype=np.float64)
    weights[y == 1] = n / (2.0 * positives)
    weights[y == 0] = n / (2.0 * negatives)
    return weights


def fit_rankers(X: np.ndarray, y: np.ndarray, candidate_rank: np.ndarray) -> dict[str, RankerBundle]:
    """Fit deliberately small candidate-ranking baselines.

    M0 is the existing raw candidate order and has no learned parameters.
    M1 is a standardized logistic regression.
    M2 is a compact histogram gradient-boosted tree model.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised by CLI dependency check
        raise RuntimeError("Install ML dependencies with: pip install -e '.[ml]'") from exc

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    _ = np.asarray(candidate_rank)

    m1 = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=400,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    m1.fit(X, y)

    m2 = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=140,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )
    m2.fit(X, y, sample_weight=_balanced_weights(y))

    return {
        "M0_raw_rank": RankerBundle("M0_raw_rank", None),
        "M1_logistic": RankerBundle("M1_logistic", m1),
        "M2_hist_gb": RankerBundle("M2_hist_gb", m2),
    }


def score_ranker(bundle: RankerBundle, X: np.ndarray, candidate_rank: np.ndarray) -> np.ndarray:
    if bundle.model is None:
        return -np.asarray(candidate_rank, dtype=np.float64)
    probabilities = bundle.model.predict_proba(np.asarray(X, dtype=np.float32))
    return np.asarray(probabilities[:, 1], dtype=np.float64)


def fit_bbox_ridge(X: np.ndarray, targets: np.ndarray):
    """Fit a cheap four-output candidate-relative bounding-box regressor."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install ML dependencies with: pip install -e '.[ml]'") from exc

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=2.0)),
        ]
    )
    model.fit(np.asarray(X, dtype=np.float32), np.asarray(targets, dtype=np.float32))
    return model
