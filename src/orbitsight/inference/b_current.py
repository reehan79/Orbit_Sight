"""Behaviour-preserving B_CURRENT inference (reference + fast).

B_CURRENT = ExtraTrees Top-1 + C4_MEDIAN centre + S2 ExtraTrees size
(with S2 features extracted around C1 centre).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from orbitsight.features import (
    extract_candidate_features,
    extract_local_geometry_features,
    refine_c1_centroid,
    refine_c4_median,
)
from orbitsight.features.candidate_features_fast import extract_candidate_features_fast
from orbitsight.features.local_geometry import roi_mask
from orbitsight.models import RankerBundle, score_ranker
from orbitsight.proposals import Candidate, RawGridProposer, infer_sensor_geometry

PRIOR_MS = 80
TOP_K = 20


@dataclass
class BCurrentResult:
    selected: Candidate | None
    features: np.ndarray
    scores: np.ndarray
    cx: float
    cy: float
    width: float
    height: float
    candidates: list[Candidate]


@dataclass
class TimedComponents:
    event_slice_ns: int = 0
    propose_ns: int = 0
    features_ns: int = 0
    ranker_ns: int = 0
    c1_ns: int = 0
    c4_ns: int = 0
    local_geom_ns: int = 0
    s2_ns: int = 0
    decode_ns: int = 0
    total_ns: int = 0


class SequenceStream:
    """Cache timestamps / geometry / proposer once per sequence."""

    def __init__(self, sequence: str, split_dir: Path):
        self.sequence = sequence
        self.npy_path = split_dir / f"{sequence}_labeled_events.npy"
        self.arr = np.load(self.npy_path, mmap_mode="r")
        self.timestamps = np.asarray(self.arr[:, 3])
        self.width, self.height, self.cell = infer_sensor_geometry(sequence)
        self.proposer = RawGridProposer(self.width, self.height, self.cell, top_k=TOP_K)
        self.sensor = (
            "DAVIS"
            if sequence.upper().startswith("DAVIS")
            else ("DVX" if sequence.upper().startswith("DVX") else "EVK4")
        )

    def slice_window(self, start_us: int, end_us: int) -> tuple[np.ndarray, np.ndarray]:
        left = int(np.searchsorted(self.timestamps, start_us, side="left"))
        right = int(np.searchsorted(self.timestamps, end_us, side="left"))
        prior_left = int(np.searchsorted(self.timestamps, start_us - PRIOR_MS * 1000, side="left"))
        current = np.asarray(self.arr[left:right, :4])
        prior = np.asarray(self.arr[prior_left:left, :4])
        return current, prior


def _s2_predict(size_trees, feat15: np.ndarray, local18: np.ndarray, cell: float) -> tuple[float, float]:
    log_wh = size_trees.predict(np.concatenate([feat15, local18]).reshape(1, -1))[0]
    return math.exp(float(log_wh[0])) * cell, math.exp(float(log_wh[1])) * cell


def run_b_current_reference(
    stream: SequenceStream,
    start_us: int,
    end_us: int,
    ranker: RankerBundle,
    size_trees,
) -> BCurrentResult:
    current, prior = stream.slice_window(start_us, end_us)
    candidates = stream.proposer.propose(current)
    if not candidates:
        return BCurrentResult(None, np.empty((0, 15), np.float32), np.empty(0), 0.0, 0.0, 0.0, 0.0, [])
    features = extract_candidate_features(
        current, prior, candidates, stream.width, stream.height, stream.cell
    )
    ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
    scores = score_ranker(ranker, features, ranks)
    sel_idx = int(np.argmax(scores))
    selected = candidates[sel_idx]
    feat15 = features[sel_idx]
    cx, cy = refine_c4_median(current, selected.cx, selected.cy, stream.cell)
    c1x, c1y = refine_c1_centroid(current, selected.cx, selected.cy, stream.cell)
    local18 = extract_local_geometry_features(
        current, c1x, c1y, stream.cell, stream.width, stream.height
    )
    w, h = _s2_predict(size_trees, feat15, local18, float(stream.cell))
    return BCurrentResult(selected, features, scores, cx, cy, w, h, candidates)


def run_b_current_fast(
    stream: SequenceStream,
    start_us: int,
    end_us: int,
    ranker: RankerBundle,
    size_trees,
) -> BCurrentResult:
    """Same mathematical definitions as reference; shared ROI + fast features."""
    current, prior = stream.slice_window(start_us, end_us)
    candidates = stream.proposer.propose(current)
    if not candidates:
        return BCurrentResult(None, np.empty((0, 15), np.float32), np.empty(0), 0.0, 0.0, 0.0, 0.0, [])
    features = extract_candidate_features_fast(
        current, prior, candidates, stream.width, stream.height, stream.cell
    )
    ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
    scores = score_ranker(ranker, features, ranks)
    sel_idx = int(np.argmax(scores))
    selected = candidates[sel_idx]
    feat15 = features[sel_idx]

    cell = float(stream.cell)
    mask = roi_mask(current, selected.cx, selected.cy, cell)
    roi = current[mask]
    if len(roi) == 0:
        cx, cy = selected.cx, selected.cy
        c1x, c1y = selected.cx, selected.cy
    else:
        cx = float(np.median(roi[:, 0]))
        cy = float(np.median(roi[:, 1]))
        c1x = float(np.mean(roi[:, 0]))
        c1y = float(np.mean(roi[:, 1]))
    local18 = extract_local_geometry_features(
        current, c1x, c1y, cell, stream.width, stream.height
    )
    w, h = _s2_predict(size_trees, feat15, local18, cell)
    return BCurrentResult(selected, features, scores, cx, cy, w, h, candidates)


def profile_b_current_window(
    stream: SequenceStream,
    start_us: int,
    end_us: int,
    ranker: RankerBundle,
    size_trees,
) -> TimedComponents:
    t = TimedComponents()
    t0 = perf_counter_ns()

    t1 = perf_counter_ns()
    current, prior = stream.slice_window(start_us, end_us)
    t.event_slice_ns = perf_counter_ns() - t1

    t1 = perf_counter_ns()
    candidates = stream.proposer.propose(current)
    t.propose_ns = perf_counter_ns() - t1
    if not candidates:
        t.total_ns = perf_counter_ns() - t0
        return t

    t1 = perf_counter_ns()
    features = extract_candidate_features(
        current, prior, candidates, stream.width, stream.height, stream.cell
    )
    t.features_ns = perf_counter_ns() - t1

    t1 = perf_counter_ns()
    ranks = np.arange(1, len(candidates) + 1, dtype=np.int16)
    scores = score_ranker(ranker, features, ranks)
    sel_idx = int(np.argmax(scores))
    selected = candidates[sel_idx]
    feat15 = features[sel_idx]
    t.ranker_ns = perf_counter_ns() - t1

    t1 = perf_counter_ns()
    c1x, c1y = refine_c1_centroid(current, selected.cx, selected.cy, stream.cell)
    t.c1_ns = perf_counter_ns() - t1

    t1 = perf_counter_ns()
    refine_c4_median(current, selected.cx, selected.cy, stream.cell)
    t.c4_ns = perf_counter_ns() - t1

    t1 = perf_counter_ns()
    local18 = extract_local_geometry_features(
        current, c1x, c1y, stream.cell, stream.width, stream.height
    )
    t.local_geom_ns = perf_counter_ns() - t1

    t1 = perf_counter_ns()
    _s2_predict(size_trees, feat15, local18, float(stream.cell))
    t.s2_ns = perf_counter_ns() - t1

    t1 = perf_counter_ns()
    _ = float(feat15[0])  # stand-in for trivial decode bookkeeping
    t.decode_ns = perf_counter_ns() - t1

    t.total_ns = perf_counter_ns() - t0
    return t
