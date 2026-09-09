#!/usr/bin/env python3
"""ONE-SHOT sealed holdout evaluation on Testing_sets.

Requires models/final/ freeze already committed.
Do not retrain. Do not change threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from orbitsight.evaluation.tii_style import (  # noqa: E402
    aggregate_tii,
    evaluate_dirs_tii,
    load_tii_gt,
    match_and_score,
)
from orbitsight.inference.b_current import SequenceStream  # noqa: E402
from orbitsight.inference.p1_detector import (  # noqa: E402
    build_gate_features,
    emit_tii_row,
    run_p1_window_fast,
)
from orbitsight.inference.windows import WINDOW_US, enumerate_challenge_windows  # noqa: E402
from orbitsight.submission import (  # noqa: E402
    discover_sequences,
    load_final_bundle,
    score_gate,
    sequence_dir_for,
    write_sequence_prediction,
)

DEFAULT_TEST = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Testing_sets"
)
TII_EVAL = Path(
    r"D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_DataLoader\evaluate.py"
)


def sensor_name(sequence: str) -> str:
    u = sequence.upper()
    if u.startswith("DAVIS"):
        return "DAVIS"
    if u.startswith("DVX"):
        return "DVX"
    return "EVK4"


def run_tii_official(gt_dir: Path, pred_dir: Path, excel_out: Path) -> dict[str, float]:
    cmd = [
        sys.executable,
        str(TII_EVAL),
        "--gt-dir",
        str(gt_dir),
        "--pred-dir",
        str(pred_dir),
        "--iou",
        "0.5",
        "--excel-out",
        str(excel_out.resolve()),
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        cmd,
        check=False,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(text.encode("ascii", errors="replace").decode("ascii"), flush=True)
    if proc.returncode != 0 and not excel_out.exists():
        raise RuntimeError(f"TII exit={proc.returncode}")
    out: dict[str, float] = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("| Precision"):
            out["precision"] = float([p.strip() for p in s.split("|") if p.strip()][1])
        elif s.startswith("| Recall"):
            out["recall"] = float([p.strip() for p in s.split("|") if p.strip()][1])
        elif s.startswith("| F1 Score"):
            out["f1"] = float([p.strip() for p in s.split("|") if p.strip()][1])
        elif "mAP @ IoU 0.5" in s:
            out["map50"] = float([p.strip() for p in s.split("|") if p.strip()][1])
    return out


def infer_with_latency(sequence: str, split_dir: Path, bundle, max_windows: int = 10**9):
    stream = SequenceStream(sequence, split_dir)
    rows = []
    samples_ms = []
    n = 0
    for ws in enumerate_challenge_windows(stream.timestamps):
        if n >= max_windows:
            break
        we = int(ws) + WINDOW_US
        t0 = perf_counter_ns()
        res = run_p1_window_fast(
            stream, int(ws), we, bundle.conf_model, bundle.size_trees, always_emit=True
        )
        if res is not None:
            gf = build_gate_features(res, stream, bundle.size_trees, reuse_geometry=True)
            gate_p = score_gate(bundle, gf)
            if gate_p >= bundle.threshold:
                rows.append(emit_tii_row(res, confidence=gate_p))
        samples_ms.append((perf_counter_ns() - t0) / 1e6)
        n += 1
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5]))
    return rows, samples_ms


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "final")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "runs" / "2026-09-08" / "final_sealed_test",
    )
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "runs" / "2026-09-08_FINAL_SEALED_TEST.md")
    args = parser.parse_args()

    test_dir = args.test_dir.resolve()
    if "Testing_sets" not in str(test_dir):
        raise SystemExit("STOP: sealed test must point at Testing_sets")
    if not (args.model_dir / "manifest.json").exists():
        raise SystemExit("STOP: models/final missing — freeze first")

    lock = args.out_dir / "SEALED_RUN_LOCK.json"
    if lock.exists():
        raise SystemExit(f"STOP: sealed test already run ({lock}). Do not re-run.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_final_bundle(args.model_dir)
    sequences = discover_sequences(test_dir)
    if len(sequences) != 4:
        raise SystemExit(f"STOP: expected 4 test sequences, found {len(sequences)}: {sequences}")

    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    latency_by_sensor: dict[str, list[float]] = defaultdict(list)
    all_latency: list[float] = []
    per_seq_rows = []

    for sequence in sequences:
        split_dir = sequence_dir_for(test_dir, sequence)
        print(f"Infer {sequence}...", flush=True)
        rows, samples = infer_with_latency(sequence, split_dir, bundle)
        write_sequence_prediction(pred_dir, sequence, rows)
        sensor = sensor_name(sequence)
        latency_by_sensor[sensor].extend(samples)
        all_latency.extend(samples)
        gt = load_tii_gt(split_dir / f"{sequence}_bb_windows_40ms.txt")
        m = match_and_score(gt, rows)
        per_seq_rows.append(
            {
                "sequence": sequence,
                "sensor": sensor,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "ap50": m.ap50,
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
                "mean_matched_iou": m.mean_matched_iou,
                "n_pred": len(rows),
                "latency_n": len(samples),
                "latency_p50_ms": float(np.percentile(samples, 50)) if samples else None,
                "latency_p95_ms": float(np.percentile(samples, 95)) if samples else None,
                "latency_p99_ms": float(np.percentile(samples, 99)) if samples else None,
            }
        )

    # Local evaluator via dirs
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        gt_dir = td_path / "gt"
        gt_dir.mkdir()
        for sequence in sequences:
            split_dir = sequence_dir_for(test_dir, sequence)
            shutil.copy2(
                split_dir / f"{sequence}_bb_windows_40ms.txt",
                gt_dir / f"{sequence}_bb_windows_40ms.txt",
            )
        local = aggregate_tii(evaluate_dirs_tii(gt_dir, pred_dir))
        excel = args.out_dir / "tii_FINAL_SEALED.xlsx"
        tii = run_tii_official(gt_dir, pred_dir, excel)

    for name in ("precision", "recall", "f1"):
        if abs(float(getattr(local, name)) - float(tii[name])) > 1e-4:
            raise SystemExit(f"STOP parity {name} local={getattr(local, name)} tii={tii[name]}")
    if abs(float(local.ap50) - float(tii["map50"])) > 1e-4:
        raise SystemExit(f"STOP parity ap50 local={local.ap50} tii={tii['map50']}")

    # Sensor aggregates
    sensor_rows = []
    by_sensor_metrics: dict[str, list] = defaultdict(list)
    for r in per_seq_rows:
        by_sensor_metrics[r["sensor"]].append(r["sequence"])
    for sensor, seqs in sorted(by_sensor_metrics.items()):
        preds = {}
        gts = {}
        for sequence in seqs:
            split_dir = sequence_dir_for(test_dir, sequence)
            gts[sequence] = load_tii_gt(split_dir / f"{sequence}_bb_windows_40ms.txt")
            # reload preds from file via match already have per-seq; recompute pooled
        # Use per_seq aggregation
        subset = [r for r in per_seq_rows if r["sensor"] == sensor]
        tp = sum(r["tp"] for r in subset)
        fp = sum(r["fp"] for r in subset)
        fn = sum(r["fn"] for r in subset)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        lat = latency_by_sensor[sensor]
        sensor_rows.append(
            {
                "sensor": sensor,
                "n_sequences": len(subset),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "latency_p50_ms": float(np.percentile(lat, 50)) if lat else None,
                "latency_p95_ms": float(np.percentile(lat, 95)) if lat else None,
                "latency_p99_ms": float(np.percentile(lat, 99)) if lat else None,
            }
        )

    overall_lat = {
        "p50_ms": float(np.percentile(all_latency, 50)),
        "p95_ms": float(np.percentile(all_latency, 95)),
        "p99_ms": float(np.percentile(all_latency, 99)),
        "n": len(all_latency),
    }

    summary = {
        "label": "FINAL SEALED HOLDOUT — NO POST-TEST TUNING",
        "model_dir": str(args.model_dir),
        "d2_threshold": bundle.threshold,
        "git_sha_models": bundle.manifest.get("git_sha"),
        "sequences": sequences,
        "local": {
            "precision": local.precision,
            "recall": local.recall,
            "f1": local.f1,
            "ap50": local.ap50,
            "tp": local.tp,
            "fp": local.fp,
            "fn": local.fn,
            "mean_matched_iou": local.mean_matched_iou,
        },
        "tii": tii,
        "latency_overall": overall_lat,
        "parity_ok": True,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out_dir / "by_sequence.csv", per_seq_rows)
    write_csv(args.out_dir / "by_sensor.csv", sensor_rows)
    lock.write_text(
        json.dumps({"sealed": True, "sequences": sequences, "f1": local.f1}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Markdown report
    lines = [
        "# FINAL SEALED HOLDOUT — NO POST-TEST TUNING",
        "",
        f"Date: 2026-09-08",
        f"Model dir: `models/final/` (threshold={bundle.threshold:.12f})",
        f"Model train git SHA (manifest): `{bundle.manifest.get('git_sha')}`",
        f"Test sequences: {len(sequences)} (Testing_sets only)",
        "",
        "**This is the one and only sealed holdout run. Do not change the model.**",
        "",
        "## Overall (local evaluator; TII parity required)",
        "",
        "| metric | local | TII |",
        "|--------|------:|----:|",
        f"| precision | {local.precision:.4f} | {tii['precision']:.4f} |",
        f"| recall | {local.recall:.4f} | {tii['recall']:.4f} |",
        f"| F1 | {local.f1:.4f} | {tii['f1']:.4f} |",
        f"| AP@0.5 | {local.ap50:.4f} | {tii['map50']:.4f} |",
        f"| TP / FP / FN | {local.tp} / {local.fp} / {local.fn} | — |",
        f"| mean matched IoU | {local.mean_matched_iou:.4f} | — |",
        "",
        "## Per sequence",
        "",
        "| sequence | sensor | P | R | F1 | AP@0.5 | TP | FP | FN | mean IoU | p95 ms |",
        "|----------|--------|--:|--:|---:|------:|---:|---:|---:|---------:|-------:|",
    ]
    for r in per_seq_rows:
        lines.append(
            f"| {r['sequence']} | {r['sensor']} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | {r['ap50']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['mean_matched_iou']:.4f} | {r['latency_p95_ms']:.2f} |"
        )
    lines += [
        "",
        "## Per sensor",
        "",
        "| sensor | P | R | F1 | TP | FP | FN | p50 ms | p95 ms | p99 ms |",
        "|--------|--:|--:|---:|---:|---:|---:|-------:|-------:|-------:|",
    ]
    for r in sensor_rows:
        lines.append(
            f"| {r['sensor']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | {r['latency_p50_ms']:.2f} | "
            f"{r['latency_p95_ms']:.2f} | {r['latency_p99_ms']:.2f} |"
        )
    lines += [
        "",
        "## Runtime overall",
        "",
        f"| p50 | p95 | p99 | n |",
        f"|----:|----:|----:|--:|",
        f"| {overall_lat['p50_ms']:.2f} | {overall_lat['p95_ms']:.2f} | {overall_lat['p99_ms']:.2f} | {overall_lat['n']} |",
        "",
        "Artifacts: `docs/runs/2026-09-08/final_sealed_test/`",
        "",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["local"], indent=2), flush=True)
    print("SEALED TEST COMPLETE", flush=True)


if __name__ == "__main__":
    main()
