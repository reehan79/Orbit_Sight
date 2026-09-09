"""python -m orbitsight.submission

Competition entrypoint: frozen D2 inference + official public packaging.
No training, no GT influence on detections, no temporal rescue, no internet.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from orbitsight.submission import DEFAULT_TEAM, run_dataset
from orbitsight.submission.public_contract import DEFAULT_CLASS_ID, DEFAULT_FILENAME_MODE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OrbitSight frozen D2 submission inference")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("ORBITSIGHT_DATASET", "/OrbitSight_dataset")),
        help="Read-only dataset root containing *_labeled_events.npy",
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=Path(os.environ.get("ORBITSIGHT_WORK", "/work")),
        help="Writable output root; writes /work/<TEAM>/<DDMMYYYY>/",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory with final D2 joblibs + manifest (default: bundled models/final)",
    )
    parser.add_argument(
        "--team",
        type=str,
        default=os.environ.get("ORBITSIGHT_TEAM_NAME", DEFAULT_TEAM),
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="Output date stamp DDMMYYYY (default: UTC today)",
    )
    parser.add_argument(
        "--filename-mode",
        type=str,
        default=os.environ.get("ORBITSIGHT_PRED_FILENAME_MODE", DEFAULT_FILENAME_MODE),
        help="admin_pred=<seq>_pred.txt | public_plain=<seq>.txt",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=int(os.environ.get("ORBITSIGHT_CLASS_ID", str(DEFAULT_CLASS_ID))),
        help="Official class_id column (default 1 = RSO per dataset docs)",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default=os.environ.get("ORBITSIGHT_SPLIT_MODE", "auto"),
        help="auto|testing|training|both (default auto; both pending organizer clarification)",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip Evaluation_Metrics.xlsx (predictions still written)",
    )
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    out = run_dataset(
        args.dataset,
        args.work,
        model_dir=args.model_dir,
        team_name=args.team,
        day=args.day,
        use_fast=True,
        filename_mode=args.filename_mode,
        class_id=args.class_id,
        split_mode=args.split_mode,
        write_metrics=not args.no_metrics,
    )
    print(f"DONE output={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
