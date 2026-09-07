"""Parity tests for static S2 feature cache (no full CV)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orbitsight.sprint.s2_static_cache import (
    build_s2_rows_for_sequence,
    build_s2_xy_dynamic,
)


SPLIT = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
)
TABLE = Path("artifacts/candidate_table.csv")

REPR = [
    "DAVIS_EGS_16908_2024-11-01-19-10-44",
    "DVX_Filtered_Stars_2025-01-20-19-15-10",
    "2025_12_23_21_12_28_EVK4_mag5.2",
]


def _load_table():
    import sys

    sys.path.insert(0, "scripts")
    import cv_challenge_aligned_confidence as cv

    return cv.load_table(TABLE)


@pytest.mark.skipif(not TABLE.exists() or not SPLIT.exists(), reason="local artifacts/data required")
def test_s2_static_cache_parity_representative_sensors(tmp_path: Path):
    table = _load_table()
    for sequence in REPR:
        if not (SPLIT / f"{sequence}_labeled_events.npy").exists():
            pytest.skip(f"missing sequence {sequence}")
        mask = np.flatnonzero(table["sequence"] == sequence)
        if len(mask) == 0:
            pytest.skip(f"no table rows for {sequence}")
        X_dyn, y_dyn = build_s2_xy_dynamic(table, mask, SPLIT)
        X_c, y_c = build_s2_rows_for_sequence(sequence, table, SPLIT)
        assert X_dyn.shape == X_c.shape
        assert y_dyn.shape == y_c.shape
        if len(X_dyn) == 0:
            continue
        # Order within sequence should match
        np.testing.assert_allclose(X_dyn, X_c, atol=1e-6, rtol=0)
        np.testing.assert_allclose(y_dyn, y_c, atol=1e-6, rtol=0)
