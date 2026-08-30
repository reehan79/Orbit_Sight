import numpy as np

from orbitsight.features.local_geometry import extract_local_geometry_features, refine_c1_centroid


def test_refine_c1_centroid_uses_roi_mean():
    events = np.array([[10.0, 10.0, 1.0, 0.0], [12.0, 12.0, 0.0, 1.0]], dtype=np.float64)
    cx, cy = refine_c1_centroid(events, 10.0, 10.0, cell=4.0)
    assert cx == 11.0
    assert cy == 11.0


def test_local_geometry_feature_length():
    events = np.array([[10.0, 10.0, 1.0, 0.0], [12.0, 12.0, 0.0, 1.0]], dtype=np.float64)
    feat = extract_local_geometry_features(events, 10.0, 10.0, cell=4.0, width=32, height=32)
    assert feat.shape == (18,)
    assert feat[-1] == 0.5
