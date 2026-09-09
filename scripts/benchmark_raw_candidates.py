from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from orbitsight.benchmark import benchmark
from orbitsight.proposals import RawGridProposer, infer_sensor_geometry

WINDOW_US = 40_000


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU benchmark for the raw sparse proposer")
    parser.add_argument("--npy", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--windows", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    arr = np.load(Path(args.npy), mmap_mode="r")
    width, height, cell = infer_sensor_geometry(args.sequence)
    proposer = RawGridProposer(width, height, cell, top_k=args.top_k)
    timestamps = arr[:, 3]
    base = int(timestamps[0])
    windows = []
    for i in range(args.windows):
        start = base + i * WINDOW_US
        end = start + WINDOW_US
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="left"))
        if right > left:
            windows.append(np.asarray(arr[left:right, :4]))
    stats = benchmark(proposer.propose, windows)
    print(f"calls={stats.calls}")
    print(f"mean_ms={stats.mean_ms:.4f}")
    print(f"p50_ms={stats.p50_ms:.4f}")
    print(f"p95_ms={stats.p95_ms:.4f}")
    print(f"p99_ms={stats.p99_ms:.4f}")


if __name__ == "__main__":
    main()
