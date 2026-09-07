"""Local-only static S2 training feature cache (execution surgery; math unchanged)."""

from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from orbitsight.features import extract_local_geometry_features, refine_c1_centroid
from orbitsight.proposals import RawGridProposer, infer_sensor_geometry

TOP_K = 20
S2_STATIC_DIR = Path("artifacts/temporal_rescue_static/s2")


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def log_rss(label: str) -> None:
    rss = _rss_mb()
    if rss is not None:
        print(f"  RSS {label}: {rss:.1f} MB", flush=True)


def s2_cache_path(sequence: str, cache_dir: Path = S2_STATIC_DIR) -> Path:
    return cache_dir / f"{sequence}.npz"


def build_s2_rows_for_sequence(
    sequence: str,
    table: dict,
    split_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic positive-row S2 X (33-d) and y for one sequence (label-free feats)."""
    seq_mask = table["sequence"] == sequence
    pos = np.flatnonzero(seq_mask & (table["target"] == 1))
    if len(pos) == 0:
        return np.empty((0, 33), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    by_w: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in pos:
        by_w[(int(table["start"][idx]), int(table["end"][idx]))].append(int(idx))

    arr = np.load(split_dir / f"{sequence}_labeled_events.npy", mmap_mode="r")
    ts = arr[:, 3]
    width, height, cell = infer_sensor_geometry(sequence)
    proposer = RawGridProposer(width, height, cell, top_k=TOP_K)

    X_rows: list[np.ndarray] = []
    y_rows: list[list[float]] = []
    for (start, end), indices in by_w.items():
        left = int(np.searchsorted(ts, start, side="left"))
        right = int(np.searchsorted(ts, end, side="left"))
        current = np.asarray(arr[left:right, :4])
        cands = proposer.propose(current)
        if not cands:
            continue
        for idx in indices:
            rank = int(table["rank"][idx])
            if rank < 1 or rank > len(cands):
                continue
            cand = cands[rank - 1]
            rcx, rcy = refine_c1_centroid(current, cand.cx, cand.cy, cell)
            local18 = extract_local_geometry_features(current, rcx, rcy, cell, width, height)
            X_rows.append(np.concatenate([table["X"][idx], local18]))
            y_rows.append(table["bbox_log_wh"][idx].tolist())

    if not X_rows:
        return np.empty((0, 33), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.asarray(X_rows, np.float32), np.asarray(y_rows, np.float32)


def ensure_s2_sequence_cache(
    sequence: str,
    table: dict,
    split_dir: Path,
    cache_dir: Path = S2_STATIC_DIR,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = s2_cache_path(sequence, cache_dir)
    if path.exists():
        return path
    X, y = build_s2_rows_for_sequence(sequence, table, split_dir)
    np.savez_compressed(path, X=X, y=y, sequence=np.asarray([sequence], dtype=object))
    return path


def load_s2_xy_for_sequences(
    sequences: list[str],
    table: dict,
    split_dir: Path,
    cache_dir: Path = S2_STATIC_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached S2 rows for TRAIN sequences only (no val mix-in)."""
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for sequence in sequences:
        path = ensure_s2_sequence_cache(sequence, table, split_dir, cache_dir)
        data = np.load(path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        if len(X) == 0:
            continue
        Xs.append(X)
        ys.append(y)
    if not Xs:
        return np.empty((0, 33), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def fit_size_s2_cached(table, train_idx, split_dir: Path, cache_dir: Path = S2_STATIC_DIR):
    """Same ExtraTreesRegressor as fit_size_s2; X/y from static per-sequence cache."""
    train_seqs = sorted({str(s) for s in table["sequence"][train_idx]})
    X, y = load_s2_xy_for_sequences(train_seqs, table, split_dir, cache_dir)
    model = ExtraTreesRegressor(
        n_estimators=32,
        max_depth=12,
        min_samples_leaf=24,
        max_features=None,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def build_s2_xy_dynamic(table, train_idx, split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Reference dynamic builder matching cv_challenge_aligned_confidence.fit_size_s2."""
    X_rows, y_rows = [], []
    pos = train_idx[table["target"][train_idx] == 1]
    by_w: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for idx in pos:
        by_w[
            (str(table["sequence"][idx]), int(table["start"][idx]), int(table["end"][idx]))
        ].append(int(idx))
    for (sequence, start, end), indices in by_w.items():
        arr = np.load(split_dir / f"{sequence}_labeled_events.npy", mmap_mode="r")
        ts = arr[:, 3]
        width, height, cell = infer_sensor_geometry(sequence)
        left = int(np.searchsorted(ts, start, side="left"))
        right = int(np.searchsorted(ts, end, side="left"))
        current = np.asarray(arr[left:right, :4])
        proposer = RawGridProposer(width, height, cell, top_k=TOP_K)
        cands = proposer.propose(current)
        if not cands:
            continue
        for idx in indices:
            rank = int(table["rank"][idx])
            if rank < 1 or rank > len(cands):
                continue
            cand = cands[rank - 1]
            rcx, rcy = refine_c1_centroid(current, cand.cx, cand.cy, cell)
            local18 = extract_local_geometry_features(current, rcx, rcy, cell, width, height)
            X_rows.append(np.concatenate([table["X"][idx], local18]))
            y_rows.append(table["bbox_log_wh"][idx].tolist())
    if not X_rows:
        return np.empty((0, 33), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.asarray(X_rows, np.float32), np.asarray(y_rows, np.float32)


def gc_release(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
