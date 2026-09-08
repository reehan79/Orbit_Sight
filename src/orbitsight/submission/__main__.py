"""python -m orbitsight.submission

Competition entrypoint: frozen D2 inference only.
No training, no GT dependency, no temporal rescue, no internet.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from orbitsight.submission import DEFAULT_TEAM, run_dataset


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
    )
    print(f"DONE output={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
