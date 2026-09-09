"""Frozen D2 competition submission runtime (no temporal rescue).

Detection science is frozen. Output packaging follows official public-timeline
contract V2 (see orbitsight.submission.public_contract).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from orbitsight.inference.b_current import SequenceStream
from orbitsight.inference.p1_detector import (
    build_gate_features,
    emit_tii_row,
    run_p1_window_fast,
    run_p1_window_reference,
)
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows
from orbitsight.submission.metrics_xlsx import (
    find_gt_path,
    score_sequence,
    write_evaluation_metrics_xlsx,
)
from orbitsight.submission.public_contract import (
    DEFAULT_CLASS_ID,
    prediction_filename,
    resolved_class_id,
    resolved_filename_mode,
    write_public_prediction_file,
)

DEFAULT_TEAM = "SparseSight-SSA"
EVENT_STEM = re.compile(r"^(.+)_labeled_events\.npy$")

# Split selection (organizer follow-up pending on emitting both when both exist).
# Default remains historical auto: Testing_sets if present, else Training_sets, else direct.
SPLIT_MODE_AUTO = "auto"
SPLIT_MODE_TESTING = "testing"
SPLIT_MODE_TRAINING = "training"
SPLIT_MODE_BOTH = "both"
DEFAULT_SPLIT_MODE = SPLIT_MODE_AUTO
SPLIT_BEHAVIOR_DOC = (
    "Default ORBITSIGHT_SPLIT_MODE=auto: prefer Testing_sets when it contains events; "
    "else Training_sets; else direct *_labeled_events.npy under dataset root. "
    "Mode 'both' emits Training_sets then Testing_sets when both exist (parameterized; "
    "not default until organizer clarifies)."
)


@dataclass
class FinalD2Bundle:
    conf_model: object
    size_trees: object
    gate_scaler: object
    gate_clf: object
    threshold: float
    manifest: dict
    model_dir: Path


@dataclass(frozen=True)
class SequenceJob:
    sequence: str
    split_dir: Path
    label: str  # "" | "Training" | "Testing"


def default_model_dir() -> Path:
    env = os.environ.get("ORBITSIGHT_MODEL_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "models" / "final",
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
    seqs: list[str] = []
    for p in sorted(split_dir.glob("*_labeled_events.npy")):
        m = EVENT_STEM.match(p.name)
        if m:
            seqs.append(m.group(1))
    return seqs


def resolved_split_mode(override: str | None = None) -> str:
    mode = (override or os.environ.get("ORBITSIGHT_SPLIT_MODE") or DEFAULT_SPLIT_MODE).strip().lower()
    allowed = {SPLIT_MODE_AUTO, SPLIT_MODE_TESTING, SPLIT_MODE_TRAINING, SPLIT_MODE_BOTH}
    if mode not in allowed:
        raise ValueError(f"unknown ORBITSIGHT_SPLIT_MODE={mode!r}; expected one of {sorted(allowed)}")
    return mode


def resolve_split_dir(dataset_root: Path, *, split_mode: str | None = None) -> Path:
    """Deterministic single-split resolution (auto/testing/training).

    For mode 'both', raises — use list_sequence_jobs instead.
    """
    mode = resolved_split_mode(split_mode)
    if mode == SPLIT_MODE_BOTH:
        raise ValueError("resolve_split_dir does not support split_mode=both; use list_sequence_jobs")

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    direct = _sequences_in_dir(root)
    if direct and mode == SPLIT_MODE_AUTO:
        return root.resolve()

    testing = root / "Testing_sets"
    training = root / "Training_sets"
    testing_seqs = _sequences_in_dir(testing) if testing.is_dir() else []
    training_seqs = _sequences_in_dir(training) if training.is_dir() else []

    if mode == SPLIT_MODE_TESTING:
        if not testing_seqs:
            raise FileNotFoundError(f"Testing_sets empty/missing under {root}")
        return testing.resolve()
    if mode == SPLIT_MODE_TRAINING:
        if not training_seqs:
            raise FileNotFoundError(f"Training_sets empty/missing under {root}")
        return training.resolve()

    # auto
    if direct:
        return root.resolve()
    if testing_seqs:
        return testing.resolve()
    if training_seqs:
        return training.resolve()
    raise FileNotFoundError(
        f"no resolvable split under {root}: expected either direct "
        "*_labeled_events.npy, or Testing_sets/, or Training_sets/ only"
    )


def list_sequence_jobs(dataset_root: Path, *, split_mode: str | None = None) -> list[SequenceJob]:
    """Build ordered inference jobs without mixing nests unexpectedly."""
    mode = resolved_split_mode(split_mode)
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    if mode == SPLIT_MODE_BOTH:
        jobs: list[SequenceJob] = []
        training = root / "Training_sets"
        testing = root / "Testing_sets"
        for seq in _sequences_in_dir(training) if training.is_dir() else []:
            jobs.append(SequenceJob(seq, training.resolve(), "Training"))
        for seq in _sequences_in_dir(testing) if testing.is_dir() else []:
            jobs.append(SequenceJob(seq, testing.resolve(), "Testing"))
        if not jobs:
            # fall back to direct root
            for seq in _sequences_in_dir(root):
                jobs.append(SequenceJob(seq, root.resolve(), ""))
        if not jobs:
            raise FileNotFoundError(f"no sequences under {root} for split_mode=both")
        return jobs

    split = resolve_split_dir(root, split_mode=mode)
    label = ""
    if split.name == "Training_sets":
        label = "Training"
    elif split.name == "Testing_sets":
        label = "Testing"
    return [SequenceJob(seq, split, label) for seq in _sequences_in_dir(split)]


def discover_sequences(dataset_root: Path, *, split_mode: str | None = None) -> list[str]:
    jobs = list_sequence_jobs(dataset_root, split_mode=split_mode)
    # Preserve order; unique by first occurrence (both mode may not overlap).
    seen: set[str] = set()
    out: list[str] = []
    for j in jobs:
        if j.sequence not in seen:
            seen.add(j.sequence)
            out.append(j.sequence)
    if not out:
        raise FileNotFoundError(f"no *_labeled_events.npy under {dataset_root}")
    return out


def sequence_dir_for(dataset_root: Path, sequence: str, *, split_mode: str | None = None) -> Path:
    for job in list_sequence_jobs(dataset_root, split_mode=split_mode):
        if job.sequence == sequence:
            return job.split_dir
    raise FileNotFoundError(f"events not found for {sequence} under {dataset_root}")


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


def write_sequence_prediction(
    out_dir: Path,
    sequence: str,
    rows: list[tuple],
    *,
    filename_mode: str | None = None,
    class_id: int | None = None,
) -> Path:
    """Write official public-timeline prediction file (serialization adapter only)."""
    name = prediction_filename(sequence, mode=filename_mode)
    path = Path(out_dir) / name
    write_public_prediction_file(path, sequence, rows, class_id=class_id)
    return path


def _gt_search_roots(dataset_root: Path, split_dir: Path) -> list[Path]:
    root = Path(dataset_root)
    return [
        split_dir,
        root,
        root / "Training_sets",
        root / "Testing_sets",
    ]


def run_dataset(
    dataset_root: Path,
    work_root: Path,
    *,
    model_dir: Path | None = None,
    team_name: str | None = None,
    day: str | None = None,
    use_fast: bool = True,
    filename_mode: str | None = None,
    class_id: int | None = None,
    split_mode: str | None = None,
    write_metrics: bool = True,
) -> Path:
    bundle = load_final_bundle(model_dir)
    out = output_dir(work_root, team_name=team_name, day=day)
    mode = resolved_filename_mode(filename_mode)
    cid = resolved_class_id(class_id)
    jobs = list_sequence_jobs(dataset_root, split_mode=split_mode)
    print(
        f"split_mode={resolved_split_mode(split_mode)} n_sequences={len(jobs)} "
        f"filename_mode={mode} class_id={cid}",
        flush=True,
    )

    # Phase A: inference + prediction serialization (GT must not influence this).
    cached: list[tuple[SequenceJob, list[tuple], float]] = []
    for job in jobs:
        t0 = time.perf_counter()
        rows = infer_sequence(job.sequence, job.split_dir, bundle, use_fast=use_fast)
        wall = time.perf_counter() - t0
        write_sequence_prediction(
            out,
            job.sequence,
            rows,
            filename_mode=mode,
            class_id=cid,
        )
        print(f"wrote {job.sequence} n={len(rows)} -> {out}", flush=True)
        cached.append((job, rows, wall))

    # Phase B: metrics workbook ONLY (may read GT; never feeds back into detections).
    if write_metrics:
        _write_metrics_phase(dataset_root, out, cached)

    return out


def _write_metrics_phase(
    dataset_root: Path,
    out: Path,
    cached: list[tuple[SequenceJob, list[tuple], float]],
) -> None:
    results: list[dict] = []
    missing_gt: list[str] = []
    walls = [w for _, _, w in cached]
    for job, rows, _wall in cached:
        gt_path = find_gt_path(_gt_search_roots(dataset_root, job.split_dir), job.sequence)
        if gt_path is None:
            missing_gt.append(job.sequence)
            continue
        results.append(
            score_sequence(job.sequence, rows, gt_path, label=job.label or "")
        )

    latency = {
        "n_sequences": len(cached),
        "total_wall_s": float(sum(walls)) if walls else 0.0,
        "mean_seq_wall_s": float(np.mean(walls)) if walls else 0.0,
        "p95_seq_wall_s": float(np.percentile(walls, 95)) if walls else 0.0,
        "note": "Wall-clock of frozen D2 inference only (excludes metrics I/O)",
    }

    xlsx = out / "Evaluation_Metrics.xlsx"
    if not results:
        # Predictions already written; record unavailability without failing the run.
        note = out / "Evaluation_Metrics_UNAVAILABLE.txt"
        note.write_text(
            "Evaluation_Metrics.xlsx not generated: no ground-truth "
            f"*_bb_windows_40ms.txt found for sequences: {missing_gt or [j.sequence for j,_,_ in cached]}\n",
            encoding="utf-8",
        )
        print(f"WARN metrics unavailable -> {note}", flush=True)
        return

    write_evaluation_metrics_xlsx(xlsx, results, latency=latency)
    if missing_gt:
        print(f"WARN metrics missing GT for: {missing_gt}", flush=True)
    print(f"wrote {xlsx.name} n_scored={len(results)}", flush=True)


# Re-export packaging constants for tests/docs.
__all__ = [
    "DEFAULT_TEAM",
    "DEFAULT_CLASS_ID",
    "FinalD2Bundle",
    "SequenceJob",
    "default_model_dir",
    "load_final_bundle",
    "resolve_split_dir",
    "list_sequence_jobs",
    "discover_sequences",
    "sequence_dir_for",
    "output_dir",
    "infer_sequence",
    "write_sequence_prediction",
    "run_dataset",
    "SPLIT_BEHAVIOR_DOC",
]
