from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from orbitsight.proposals import RawGridProposer, infer_sensor_geometry

WINDOW_US = 40_000


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end CPU benchmark for one 40-ms sparse-proposal step. "
            "Unlike benchmark_raw_candidates.py, this includes timestamp search, "
            "memmap slicing and proposal generation. It excludes initial np.load()."
        )
    )
    parser.add_argument("--npy", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--windows", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    arr = np.load(Path(args.npy), mmap_mode="r")
    timestamps = arr[:, 3]
    width, height, cell = infer_sensor_geometry(args.sequence)
    proposer = RawGridProposer(width, height, cell, top_k=args.top_k)

    first_t = int(timestamps[0])
    last_t = int(timestamps[-1])
    available = max(1, int((last_t - first_t) // WINDOW_US) + 1)
    n_windows = min(args.windows, available)

    # Small warm-up of Python/Numpy paths; page faults for later memmap regions are
    # still part of the measured sequential run, which is intentional.
    for i in range(min(args.warmup, n_windows)):
        start = first_t + i * WINDOW_US
        end = start + WINDOW_US
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="left"))
        proposer.propose(np.asarray(arr[left:right, :4]))

    elapsed_ms: list[float] = []
    event_counts: list[int] = []
    candidate_counts: list[int] = []

    for i in range(n_windows):
        start = first_t + i * WINDOW_US
        end = start + WINDOW_US

        t0 = perf_counter_ns()
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="left"))
        events = np.asarray(arr[left:right, :4])
        candidates = proposer.propose(events)
        t1 = perf_counter_ns()

        elapsed_ms.append((t1 - t0) / 1_000_000.0)
        event_counts.append(right - left)
        candidate_counts.append(len(candidates))

    values = np.asarray(elapsed_ms, dtype=np.float64)
    print(f"calls={len(values)}")
    print(f"mean_ms={float(values.mean()):.4f}")
    print(f"p50_ms={_percentile(elapsed_ms, 50):.4f}")
    print(f"p95_ms={_percentile(elapsed_ms, 95):.4f}")
    print(f"p99_ms={_percentile(elapsed_ms, 99):.4f}")
    print(f"max_ms={float(values.max()):.4f}")
    print(f"mean_events_per_window={float(np.mean(event_counts)):.1f}")
    print(f"mean_candidates={float(np.mean(candidate_counts)):.2f}")
    print("scope=searchsorted+memmap_slice+raw_proposal; excludes initial np.load")


if __name__ == "__main__":
    main()
