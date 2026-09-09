#!/usr/bin/env python3
"""Train final D2 deployment models on all 17 Training_sets sequences.

Uses EXACTLY the established D2 methodology from cv_challenge_aligned_confidence.py:
- sequence-level inner OOF (KFold, shuffle=True, random_state=42)
- challenge-aligned logistic gate (G1)
- TII detection-F1 threshold from train OOF only
- final fit on all 17 sequences

Never accesses Testing_sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import cv_challenge_aligned_confidence as cv  # noqa: E402

from orbitsight.features import FEATURE_NAMES, LOCAL_GEOMETRY_NAMES
from orbitsight.inference.p1_detector import GATE_FEATURE_DIM

DEFAULT_SPLIT = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets"
)
GATE_SCALAR_NAMES = (
    "conf1",
    "conf2",
    "conf3",
    "gap12",
    "gap13",
    "mean_probs",
    "std_probs",
    "rank_frac",
    "c4_dx_cells",
    "c4_dy_cells",
    "c1_dx_cells",
    "c1_dy_cells",
    "log_w",
    "log_h",
    "log_ar",
)
GATE_FEATURE_NAMES = tuple(FEATURE_NAMES) + GATE_SCALAR_NAMES + tuple(LOCAL_GEOMETRY_NAMES)


def discover_sequences(split_dir: Path) -> list[str]:
    seqs = sorted(
        p.name.replace("_labeled_events.npy", "")
        for p in split_dir.glob("*_labeled_events.npy")
    )
    if "Testing_sets" in str(split_dir.resolve()):
        raise SystemExit("STOP: refuse Testing_sets for final training")
    if len(seqs) != 17:
        raise SystemExit(f"STOP: expected 17 training sequences, found {len(seqs)}")
    return seqs


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final frozen D2 models")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts" / "all_window_candidates.npz")
    parser.add_argument("--table", type=Path, default=ROOT / "artifacts" / "candidate_table.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models" / "final")
    args = parser.parse_args()

    split_dir = args.split_dir.resolve()
    if "Testing_sets" in str(split_dir):
        raise SystemExit("STOP: Testing_sets forbidden for final training")

    sequences = discover_sequences(split_dir)
    print(f"Training sequences ({len(sequences)}):", flush=True)
    for s in sequences:
        print(f"  {s}", flush=True)

    print("Loading candidate cache/table...", flush=True)
    cache = cv.load_cache(args.cache)
    table = cv.load_table(args.table)

    print("PART A/B: sequence-level inner OOF for D2 gate + threshold...", flush=True)
    val_recs, oof_g1, _oof_g2, train_gate_g1, _train_gate_g2 = cv.inner_oof_combined(
        sequences, cache, table, split_dir
    )
    if len(oof_g1) != len(val_recs) or len(oof_g1) == 0:
        raise SystemExit(f"STOP: bad OOF sizes oof={len(oof_g1)} val={len(val_recs)}")

    thr = cv.detection_f1_threshold_from_records(val_recs, oof_g1)
    print(f"Final D2 threshold (train OOF TII detection F1): {thr:.12f}", flush=True)

    print("PART C: fit deployment models on all 17 sequences...", flush=True)
    train_idx = np.flatnonzero(np.isin(table["sequence"], sequences))
    mask = np.array([str(s) in set(sequences) for s in cache["sequence"]], dtype=bool)
    conf_model = cv.fit_confidence(cache["features"][mask], cache["target"][mask])
    size_trees = cv.fit_size_s2(table, train_idx, split_dir)

    # Match CV fold deploy: fit G1 on accumulated inner-train gate rows from OOF
    # (same process as cv_challenge_aligned_confidence.py after inner_oof_combined).
    if not train_gate_g1:
        raise SystemExit("STOP: empty train_gate_g1 from inner OOF")
    Xg = np.stack([r.gate_features for r in train_gate_g1])
    yg = np.array([r.is_tp_if_emitted for r in train_gate_g1], dtype=np.int8)
    g1_scaler, g1_clf = cv.fit_gate_g1(Xg, yg)

    # Sanity: OOF threshold applied to OOF scores for train-side diagnostic only.
    oof_diag = cv.diag_from_records(val_recs, thr, oof_g1)
    oof_preds = cv.records_to_preds(val_recs, thr, oof_g1)
    oof_score = cv.score_preds(oof_preds, sequences, split_dir)
    print(
        f"Train-OOF diagnostic F1={oof_score['overall'].f1:.4f} "
        f"P={oof_score['overall'].precision:.4f} R={oof_score['overall'].recall:.4f} "
        f"empty_fpr={oof_diag['empty_window_false_positive_rate']:.4f}",
        flush=True,
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "confidence_extratrees.joblib": conf_model,
        "size_s2_extratrees.joblib": size_trees,
        "d2_gate_scaler.joblib": g1_scaler,
        "d2_gate_logistic.joblib": g1_clf,
    }
    hashes = {}
    for name, obj in paths.items():
        p = out / name
        joblib.dump(obj, p)
        hashes[name] = sha256_file(p)

    threshold_path = out / "d2_threshold.json"
    threshold_payload = {
        "d2_threshold": float(thr),
        "selection": "TII detection F1 on sequence-level inner-OOF gate scores (Training_sets only)",
        "n_oof_windows": int(len(val_recs)),
        "train_oof_f1": float(oof_score["overall"].f1),
        "train_oof_precision": float(oof_score["overall"].precision),
        "train_oof_recall": float(oof_score["overall"].recall),
        "train_oof_empty_fpr": float(oof_diag["empty_window_false_positive_rate"]),
    }
    threshold_path.write_text(json.dumps(threshold_payload, indent=2) + "\n", encoding="utf-8")
    hashes["d2_threshold.json"] = sha256_file(threshold_path)

    feature_path = out / "feature_names.json"
    feature_payload = {
        "candidate_features": list(FEATURE_NAMES),
        "local_geometry_features": list(LOCAL_GEOMETRY_NAMES),
        "gate_features": list(GATE_FEATURE_NAMES),
        "gate_feature_dim": GATE_FEATURE_DIM,
        "candidate_feature_dim": len(FEATURE_NAMES),
        "local_geometry_dim": len(LOCAL_GEOMETRY_NAMES),
    }
    feature_path.write_text(json.dumps(feature_payload, indent=2) + "\n", encoding="utf-8")
    hashes["feature_names.json"] = sha256_file(feature_path)

    sha = git_sha()
    manifest = {
        "name": "OrbitSight final D2 freeze",
        "version": "final-d2-v1",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": sha,
        "architecture": {
            "proposals": "RawGridProposer Top-20",
            "confidence": "ExtraTreesClassifier",
            "centre": "C4_MEDIAN (S2 features at C1)",
            "size": "S2 ExtraTreesRegressor",
            "gate": "D2 StandardScaler + LogisticRegression",
            "path": "TRUE P1 + parity-preserving geometry reuse",
        },
        "hyperparameters": {
            "confidence": {
                "n_estimators": 64,
                "max_depth": 14,
                "min_samples_leaf": 12,
                "max_features": None,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": 1,
            },
            "size_s2": {
                "n_estimators": 32,
                "max_depth": 12,
                "min_samples_leaf": 24,
                "max_features": None,
                "random_state": 42,
                "n_jobs": 1,
            },
            "d2_gate": {
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
                "scaler": "StandardScaler",
            },
            "inner_oof": {"kfold_shuffle": True, "random_state": 42, "max_splits": 5},
        },
        "seeds": {"random_state": 42},
        "d2_threshold": float(thr),
        "training_sequences": sequences,
        "n_training_sequences": len(sequences),
        "split_dir": str(split_dir),
        "artifacts": {
            "cache": str(args.cache.resolve()),
            "table": str(args.table.resolve()),
        },
        "files": hashes,
        "notes": [
            "Temporal rescue rejected for deployment; not included.",
            "Threshold selected on Training_sets OOF only; Testing_sets never used.",
            "Gate fit uses accumulated inner-train OOF rows (identical to CV deploy fit).",
        ],
        "n_gate_fit_windows": int(len(train_gate_g1)),
        "n_inner_train_gate_rows_accumulated": int(len(train_gate_g1)),
    }
    man_path = out / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Re-hash manifest after write is circular; store sibling checksums only.
    print(f"Wrote {out}", flush=True)
    print(json.dumps({"d2_threshold": thr, "git_sha": sha, "files": hashes}, indent=2), flush=True)


if __name__ == "__main__":
    main()
