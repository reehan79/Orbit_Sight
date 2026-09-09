from __future__ import annotations

import argparse
import json
from pathlib import Path
from orbitsight.validation import build_sequence_folds

SUFFIX = "_bb_windows_40ms.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic sequence-level CV folds")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--out", default="sequence_folds.json")
    args = parser.parse_args()
    train = Path(args.train_dir)
    sequences = [p.name[:-len(SUFFIX)] for p in sorted(train.glob(f"*{SUFFIX}"))]
    folds = build_sequence_folds(sequences, n_splits=args.folds, seed=args.seed)
    payload = [{"fold": f.index, "train": list(f.train), "validation": list(f.validation)} for f in folds]
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(folds)} folds to {args.out}")
    for f in folds:
        print(f"Fold {f.index}: train={len(f.train)} validation={len(f.validation)}")


if __name__ == "__main__":
    main()
