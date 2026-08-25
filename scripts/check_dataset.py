from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

SUFFIX = "_bb_windows_40ms.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural integrity check for an OrbitSight split")
    parser.add_argument("--split-dir", required=True)
    args = parser.parse_args()
    root = Path(args.split_dir)
    gt_files = sorted(root.glob(f"*{SUFFIX}"))
    if not gt_files:
        raise SystemExit(f"No GT files found in {root}")
    failures = 0
    for gt in gt_files:
        sequence = gt.name[:-len(SUFFIX)]
        npy = root / f"{sequence}_labeled_events.npy"
        if not npy.exists():
            print(f"MISSING NPY  {sequence}")
            failures += 1
            continue
        arr = np.load(npy, mmap_mode="r")
        ok_shape = arr.ndim == 2 and arr.shape[1] == 6
        monotonic = bool(len(arr) < 2 or np.all(np.diff(arr[:, 3]) >= 0))
        labels_ok = bool(np.all((arr[:, 4] == 0) | (arr[:, 4] == 1)))
        ok = ok_shape and monotonic and labels_ok
        print(f"{'OK' if ok else 'FAIL':4s}  {sequence:60s} shape={arr.shape} monotonic={monotonic} labels01={labels_ok}")
        failures += 0 if ok else 1
    print(f"\nSequences: {len(gt_files)}  Failures: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
