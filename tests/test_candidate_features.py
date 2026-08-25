import numpy as np

from orbitsight.features import FEATURE_NAMES, extract_candidate_features
from orbitsight.proposals import Candidate


def test_candidate_features_are_finite_and_label_free_shape():
    current = np.array(
        [
            [10, 10, 1, 1000],
            [11, 10, 1, 1001],
            [10, 11, 0, 1002],
            [50, 50, 1, 1003],
        ],
        dtype=np.float64,
    )
    prior = np.array([[9, 10, 1, 900], [10, 10, 1, 901]], dtype=np.float64)
    candidates = [Candidate(cx=12.0, cy=12.0, score=1.0, count=3, grid_x=1, grid_y=1)]

    features = extract_candidate_features(current, prior, candidates, width=100, height=80, cell_size=8)

    assert features.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert features[0, FEATURE_NAMES.index("local_positive_fraction")] > 0.0
