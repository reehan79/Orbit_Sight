"""Frozen D2 competition submission runtime (no temporal rescue)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from orbitsight.evaluation.tii_style import write_tii_prediction_file
from orbitsight.inference.b_current import SequenceStream
from orbitsight.inference.p1_detector import (
    build_gate_features,
    emit_tii_row,
    run_p1_window_fast,
    run_p1_window_reference,
)
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows

DEFAULT_TEAM = "OrbitSight"
EVENT_STEM = re.compile(r"^(.+)_labeled_events\.npy$")


@dataclass
class FinalD2Bundle:
    conf_model: object
    size_trees: object
    gate_scaler: object
    gate_clf: object
    threshold: float
    manifest: dict
    model_dir: Path


def default_model_dir() -> Path:
    env = os.environ.get("ORBITSIGHT_MODEL_DIR")
    if env:
        return Path(env)
    # Prefer package-adjacent models/final, then /models/final in container.
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "models" / "final",  # repo root when editable
        Path("/models/final"),
        Path("models/final"),
    ]
    for c in candidates:
        if (c / "manifest.json").exists():
            return c
    return candidates[0]


def load_final_bundle(model_dir: Path | None = None) -> FinalD2Bundle:
    d = Path(model_dir) if model_dir is not None else default_model_dir()
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    thr_obj = json.loads((d / "d2_threshold.json").read_text(encoding="utf-8"))
    return FinalD2Bundle(
        conf_model=joblib.load(d / "confidence_extratrees.joblib"),
        size_trees=joblib.load(d / "size_s2_extratrees.joblib"),
        gate_scaler=joblib.load(d / "d2_gate_scaler.joblib"),
        gate_clf=joblib.load(d / "d2_gate_logistic.joblib"),
        threshold=float(thr_obj["d2_threshold"]),
        manifest=manifest,
        model_dir=d,
    )


def _sequences_in_dir(split_dir: Path) -> list[str]:
    """List sequences from *_labeled_events.npy directly inside split_dir (non-recursive)."""
    seqs: list[str] = []
    for p in sorted(split_dir.glob("*_labeled_events.npy")):
        m = EVENT_STEM.match(p.name)
        if m:
            seqs.append(m.group(1))
    return seqs


def resolve_split_dir(dataset_root: Path) -> Path:
    """Deterministic split resolution. Never recursively mixes Training_sets + Testing_sets.

    CASE A: dataset_root directly contains *_labeled_events.npy → use dataset_root.
    CASE B: else Testing_sets/ has events → use Testing_sets ONLY.
    CASE C: else Training_sets/ has events (and Testing_sets absent/empty) → Training_sets ONLY.
    CASE D: otherwise fail loudly (no recursive mixing of unexpected nests).
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    direct = _sequences_in_dir(root)
    if direct:
        return root.resolve()

    testing = root / "Testing_sets"
    training = root / "Training_sets"
    testing_seqs = _sequences_in_dir(testing) if testing.is_dir() else []
    training_seqs = _sequences_in_dir(training) if training.is_dir() else []

    if testing_seqs:
        return testing.resolve()
    if training_seqs:
        return training.resolve()

    # Unexpected nested layout — refuse to recurse / mix.
    raise FileNotFoundError(
        f"no resolvable split under {root}: expected either direct "
        "*_labeled_events.npy, or Testing_sets/, or Training_sets/ only"
    )


def discover_sequences(dataset_root: Path) -> list[str]:
    """Discover sequences in the resolved split only (non-recursive)."""
    split = resolve_split_dir(dataset_root)
    seqs = _sequences_in_dir(split)
    if not seqs:
        raise FileNotFoundError(f"no *_labeled_events.npy under resolved split {split}")
    return sorted(set(seqs))


def sequence_dir_for(dataset_root: Path, sequence: str) -> Path:
    """Return the directory containing sequence events inside the resolved split only."""
    split = resolve_split_dir(dataset_root)
    direct = split / f"{sequence}_labeled_events.npy"
    if direct.exists():
        return split
    raise FileNotFoundError(f"events not found for {sequence} under resolved split {split}")


def output_dir(work_root: Path, team_name: str | None = None, day: str | None = None) -> Path:
    team = team_name or os.environ.get("ORBITSIGHT_TEAM_NAME", DEFAULT_TEAM)
    stamp = day or datetime.now(timezone.utc).strftime("%d%m%Y")
    out = Path(work_root) / team / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def score_gate(bundle: FinalD2Bundle, gate_features: np.ndarray) -> float:
    xs = bundle.gate_scaler.transform(gate_features.reshape(1, -1))
    return float(bundle.gate_clf.predict_proba(xs)[0, 1])


def infer_sequence(
    sequence: str,
    split_dir: Path,
    bundle: FinalD2Bundle,
    *,
    use_fast: bool = True,
) -> list[tuple]:
    """Run frozen D2 on one sequence. Uses only event columns x,y,polarity,t via SequenceStream."""
    stream = SequenceStream(sequence, split_dir)
    rows: list[tuple] = []
    run_win = run_p1_window_fast if use_fast else run_p1_window_reference
    for ws in enumerate_challenge_windows(stream.timestamps):
        we = int(ws) + WINDOW_US
        # Always emit Top-1 geometry so the gate can decide (TRUE P1 + D2 gate).
        res = run_win(
            stream,
            int(ws),
            we,
            bundle.conf_model,
            bundle.size_trees,
            threshold=None,
            always_emit=True,
        )
        if res is None:
            continue
        gf = build_gate_features(res, stream, bundle.size_trees, reuse_geometry=True)
        gate_p = score_gate(bundle, gf)
        if gate_p >= bundle.threshold:
            rows.append(emit_tii_row(res, confidence=gate_p))
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5]))
    return rows


def write_sequence_prediction(out_dir: Path, sequence: str, rows: list[tuple]) -> Path:
    path = out_dir / f"{sequence}_bb_windows_40ms.txt"
    write_tii_prediction_file(path, rows)
    return path


def run_dataset(
    dataset_root: Path,
    work_root: Path,
    *,
    model_dir: Path | None = None,
    team_name: str | None = None,
    day: str | None = None,
    use_fast: bool = True,
) -> Path:
    bundle = load_final_bundle(model_dir)
    out = output_dir(work_root, team_name=team_name, day=day)
    split_dir = resolve_split_dir(dataset_root)
    sequences = discover_sequences(dataset_root)
    print(f"resolved_split={split_dir} n_sequences={len(sequences)}", flush=True)
    for sequence in sequences:
        seq_dir = sequence_dir_for(dataset_root, sequence)
        rows = infer_sequence(sequence, seq_dir, bundle, use_fast=use_fast)
        write_sequence_prediction(out, sequence, rows)
        print(f"wrote {sequence} n={len(rows)} -> {out}", flush=True)
    return out
