"""Deployable P1 detector: confidence Top-1 + C4_MEDIAN + S2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from orbitsight.features import extract_local_geometry_features, refine_c1_centroid, refine_c4_median
from orbitsight.features.candidate_features_fast import extract_candidate_features_fast
from orbitsight.features.local_geometry import roi_mask
from orbitsight.inference.b_current import SequenceStream, _s2_predict
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows
from orbitsight.proposals import Candidate


@dataclass
class P1WindowResult:
    ws: int
    we: int
    emitted: bool
    cx: float
    cy: float
    width: float
    height: float
    confidence: float
    features: np.ndarray
    probs: np.ndarray
    sel_idx: int
    candidates: list[Candidate]
    current: np.ndarray
    prior: np.ndarray


def classical_box_from_candidate(current, cand, feat15, cell, width, height, size_trees):
    cx, cy = refine_c4_median(current, cand.cx, cand.cy, cell)
    c1x, c1y = refine_c1_centroid(current, cand.cx, cand.cy, cell)
    local18 = extract_local_geometry_features(current, c1x, c1y, cell, width, height)
    log_wh = size_trees.predict(np.concatenate([feat15, local18]).reshape(1, -1))[0]
    w = math.exp(float(log_wh[0])) * cell
    h = math.exp(float(log_wh[1])) * cell
    return cx, cy, w, h, c1x, c1y, local18, float(log_wh[0]), float(log_wh[1])


def classical_box_fast(current, cand, feat15, cell, width, height, size_trees):
    """C4+C1+S2 with shared ROI mask (same math as classical_box_from_candidate)."""
    mask = roi_mask(current, cand.cx, cand.cy, cell)
    roi = current[mask]
    if len(roi) == 0:
        cx, cy = cand.cx, cand.cy
        c1x, c1y = cand.cx, cand.cy
    else:
        cx = float(np.median(roi[:, 0]))
        cy = float(np.median(roi[:, 1]))
        c1x = float(np.mean(roi[:, 0]))
        c1y = float(np.mean(roi[:, 1]))
    local18 = extract_local_geometry_features(current, c1x, c1y, cell, width, height)
    log_wh = size_trees.predict(np.concatenate([feat15, local18]).reshape(1, -1))[0]
    w = math.exp(float(log_wh[0])) * cell
    h = math.exp(float(log_wh[1])) * cell
    return cx, cy, w, h, c1x, c1y, local18, float(log_wh[0]), float(log_wh[1])


def _top_confidence_idx(probs: np.ndarray) -> int:
    return int(np.argmax(probs))


def run_p1_window_reference(
    stream: SequenceStream,
    ws: int,
    we: int,
    conf_model,
    size_trees,
    threshold: float | None = None,
    always_emit: bool = False,
) -> P1WindowResult | None:
    """Exact P1 semantics: Top-1 confidence; geometry only for winner if emitting."""
    current, prior = stream.slice_window(ws, we)
    candidates = stream.proposer.propose(current)
    if not candidates:
        return None
    features = extract_candidate_features_fast(
        current, prior, candidates, stream.width, stream.height, stream.cell
    )
    probs = conf_model.predict_proba(features)[:, 1]
    sel_idx = _top_confidence_idx(probs)
    conf = float(probs[sel_idx])
    emit = always_emit or (threshold is not None and conf >= threshold)
    cx, cy, w, h = 0.0, 0.0, 0.0, 0.0
    if emit:
        cx, cy, w, h, _, _, _, _, _ = classical_box_from_candidate(
            current,
            candidates[sel_idx],
            features[sel_idx],
            stream.cell,
            stream.width,
            stream.height,
            size_trees,
        )
    return P1WindowResult(
        ws=int(ws),
        we=int(we),
        emitted=emit,
        cx=cx,
        cy=cy,
        width=w,
        height=h,
        confidence=conf,
        features=features,
        probs=probs,
        sel_idx=sel_idx,
        candidates=candidates,
        current=current,
        prior=prior,
    )


def run_p1_window_fast(
    stream: SequenceStream,
    ws: int,
    we: int,
    conf_model,
    size_trees,
    threshold: float | None = None,
    always_emit: bool = False,
) -> P1WindowResult | None:
    """TRUE P1 fast path: no geometry unless top-1 will be emitted."""
    current, prior = stream.slice_window(ws, we)
    candidates = stream.proposer.propose(current)
    if not candidates:
        return None
    features = extract_candidate_features_fast(
        current, prior, candidates, stream.width, stream.height, stream.cell
    )
    probs = conf_model.predict_proba(features)[:, 1]
    sel_idx = _top_confidence_idx(probs)
    conf = float(probs[sel_idx])
    emit = always_emit or (threshold is not None and conf >= threshold)
    cx, cy, w, h = 0.0, 0.0, 0.0, 0.0
    if emit:
        cx, cy, w, h, _, _, _, _, _ = classical_box_fast(
            current,
            candidates[sel_idx],
            features[sel_idx],
            stream.cell,
            stream.width,
            stream.height,
            size_trees,
        )
    return P1WindowResult(
        ws=int(ws),
        we=int(we),
        emitted=emit,
        cx=cx,
        cy=cy,
        width=w,
        height=h,
        confidence=conf,
        features=features,
        probs=probs,
        sel_idx=sel_idx,
        candidates=candidates,
        current=current,
        prior=prior,
    )


def emit_tii_row(result: P1WindowResult) -> tuple:
    return (
        int(result.ws),
        int(result.we),
        int(round(result.cx)),
        int(round(result.cy)),
        int(round(result.width)),
        int(round(result.height)),
        float(result.confidence),
    )


def build_gate_features(
    result: P1WindowResult,
    stream: SequenceStream,
    size_trees,
) -> np.ndarray:
    """Fixed gate feature vector (49 dims) for unthresholded Top-1."""
    sel = result.candidates[result.sel_idx]
    feat15 = result.features[result.sel_idx]
    cell = float(stream.cell)
    cx, cy, w, h, c1x, c1y, local18, log_w, log_h = classical_box_from_candidate(
        result.current,
        sel,
        feat15,
        stream.cell,
        stream.width,
        stream.height,
        size_trees,
    )
    order = np.argsort(-result.probs)
    p_sorted = result.probs[order]
    conf2 = float(p_sorted[1]) if len(p_sorted) > 1 else 0.0
    conf3 = float(p_sorted[2]) if len(p_sorted) > 2 else 0.0
    rank_frac = float(result.sel_idx + 1) / float(len(result.candidates))
    log_ar = abs(log_w - log_h)
    return np.concatenate(
        [
            feat15,
            np.array(
                [
                    float(result.confidence),
                    conf2,
                    conf3,
                    float(result.confidence) - conf2,
                    float(result.confidence) - conf3,
                    float(np.mean(result.probs)),
                    float(np.std(result.probs)),
                    rank_frac,
                    (cx - sel.cx) / cell,
                    (cy - sel.cy) / cell,
                    (c1x - sel.cx) / cell,
                    (c1y - sel.cy) / cell,
                    log_w,
                    log_h,
                    log_ar,
                ],
                dtype=np.float32,
            ),
            local18.astype(np.float32),
        ]
    ).astype(np.float32)


GATE_FEATURE_DIM = 49


def benchmark_p1_latency(
    stream: SequenceStream,
    conf_model,
    size_trees,
    threshold: float,
    max_windows: int = 500,
    gate_scaler=None,
    gate_clf=None,
    gate_et=None,
    gate_threshold: float = 0.5,
) -> list[float]:
    """Complete deployable P1 path latency samples (ms)."""
    samples: list[float] = []
    n = 0
    use_gate = gate_et is not None or (gate_scaler is not None and gate_clf is not None)
    for ws in enumerate_challenge_windows(stream.timestamps):
        if n >= max_windows:
            break
        we = int(ws) + WINDOW_US
        t0 = perf_counter_ns()
        if use_gate:
            # Gate needs geometry features for the Top-1 box.
            res = run_p1_window_fast(stream, int(ws), we, conf_model, size_trees, always_emit=True)
            if res is None:
                samples.append((perf_counter_ns() - t0) / 1e6)
                n += 1
                continue
            gf = build_gate_features(res, stream, size_trees)
            if gate_et is not None:
                score = float(gate_et.predict_proba(gf.reshape(1, -1))[0, 1])
            else:
                score = float(gate_clf.predict_proba(gate_scaler.transform(gf.reshape(1, -1)))[0, 1])
            if score >= gate_threshold:
                emit_tii_row(res)
        else:
            # TRUE P1: geometry only if Top-1 confidence >= threshold.
            res = run_p1_window_fast(
                stream, int(ws), we, conf_model, size_trees, threshold=threshold, always_emit=False
            )
            if res is not None and res.emitted:
                emit_tii_row(res)
        samples.append((perf_counter_ns() - t0) / 1e6)
        n += 1
    return samples
