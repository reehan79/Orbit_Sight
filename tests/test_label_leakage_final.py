"""Label-column leakage hard test for frozen D2 inference.

Predictions must be bit-identical when column 4 is randomized / all-0 / all-1.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from orbitsight.submission import FinalD2Bundle, infer_sequence, load_final_bundle


TRAIN = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
)
MODEL = Path(__file__).resolve().parents[1] / "models" / "final"
SEQ = "DAVIS_SL12RB2_15772_2024-12-04-18-21-37"


def _bundle_or_skip() -> FinalD2Bundle:
    if not (MODEL / "manifest.json").exists():
        pytest.skip("models/final not trained yet")
    return load_final_bundle(MODEL)


def _rows_equal(a: list[tuple], b: list[tuple]) -> None:
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra[:6] == rb[:6]
        assert abs(float(ra[6]) - float(rb[6])) <= 1e-12


@pytest.mark.skipif(not (TRAIN / f"{SEQ}_labeled_events.npy").exists(), reason="Training_sets missing")
def test_label_column_does_not_affect_final_inference():
    bundle = _bundle_or_skip()
    src = TRAIN / f"{SEQ}_labeled_events.npy"
    base = np.load(src)
    assert base.ndim == 2 and base.shape[1] >= 4

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # baseline
        np.save(td_path / f"{SEQ}_labeled_events.npy", base)
        base_rows = infer_sequence(SEQ, td_path, bundle, use_fast=True)

        # randomized labels
        rnd = base.copy()
        if rnd.shape[1] > 4:
            rng = np.random.default_rng(0)
            rnd[:, 4] = rng.integers(0, 2, size=len(rnd))
        np.save(td_path / f"{SEQ}_labeled_events.npy", rnd)
        rnd_rows = infer_sequence(SEQ, td_path, bundle, use_fast=True)
        _rows_equal(base_rows, rnd_rows)

        # all zeros
        z = base.copy()
        if z.shape[1] > 4:
            z[:, 4] = 0
        np.save(td_path / f"{SEQ}_labeled_events.npy", z)
        z_rows = infer_sequence(SEQ, td_path, bundle, use_fast=True)
        _rows_equal(base_rows, z_rows)

        # all ones
        o = base.copy()
        if o.shape[1] > 4:
            o[:, 4] = 1
        np.save(td_path / f"{SEQ}_labeled_events.npy", o)
        o_rows = infer_sequence(SEQ, td_path, bundle, use_fast=True)
        _rows_equal(base_rows, o_rows)


def test_runtime_imports_exclude_temporal_rescue_and_label_loaders():
    import orbitsight.submission as sub
    import orbitsight.inference.b_current as bc
    import orbitsight.inference.p1_detector as p1

    for mod in (sub, bc, p1):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "temporal_rescue" not in src
        assert "[:, 4]" not in src
        assert "arr[:,4]" not in src.replace(" ", "")
