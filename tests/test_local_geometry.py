import numpy as np

from orbitsight.features.local_geometry import (
    extract_local_geometry_features,
    local_extent_from_roi,
    refine_c1_centroid,
    refine_c4_median,
    refine_c5_soft_background_centroid,
)


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


def test_refine_c4_median():
    events = np.array([[8.0, 10.0, 1.0, 0.0], [12.0, 14.0, 0.0, 1.0]], dtype=np.float64)
    cx, cy = refine_c4_median(events, 10.0, 10.0, cell=4.0)
    assert cx == 10.0
    assert cy == 12.0


def test_local_extent_minimum_one_pixel():
    events = np.array([[10.0, 10.0, 1.0, 0.0]], dtype=np.float64)
    w, h = local_extent_from_roi(events, 10.0, 10.0, cell=4.0)
    assert w >= 1.0
    assert h >= 1.0


def test_c5_soft_background_downweights_hot_pixels():
    current = np.array([[10.0, 10.0, 1.0, 0.0], [12.0, 12.0, 1.0, 1.0]], dtype=np.float64)
    prior = np.array([[10.0, 10.0, 1.0, 0.0], [10.0, 10.0, 1.0, 1.0], [10.0, 10.0, 0.0, 2.0]], dtype=np.float64)
    cx, cy = refine_c5_soft_background_centroid(current, prior, 10.0, 10.0, cell=4.0)
    assert cx > 10.0
    assert cy > 10.0

