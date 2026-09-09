from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from orbitsight.proposals import RawGridProposer, infer_sensor_geometry

WINDOW_US = 40_000


def run(path: Path, sequence: str, top_k: int, windows: int, mode: str) -> None:
    wall0 = time.perf_counter()
    if mode == "ram":
        arr = np.load(path)
    else:
        arr = np.load(path, mmap_mode="r")
    load_ms = (time.perf_counter() - wall0) * 1000.0

    width, height, cell = infer_sensor_geometry(sequence)
    proposer = RawGridProposer(width, height, cell, top_k=top_k)

    total0 = time.perf_counter()
    timestamps = np.asarray(arr[:, 3], dtype=np.int64).copy()
    first = int(timestamps[0])
    last = int(timestamps[-1])
    max_windows = max(0, int((last - first) // WINDOW_US))
    nwin = min(windows, max_windows) if windows > 0 else max_windows
    starts = first + np.arange(nwin, dtype=np.int64) * WINDOW_US
    ends = starts + WINDOW_US
    lefts = np.searchsorted(timestamps, starts, side="left")
    rights = np.searchsorted(timestamps, ends, side="left")

    proposal_times = []
    calls = 0
    total_events = 0
    for left, right in zip(lefts, rights):
        if right <= left:
            continue
        t0 = time.perf_counter()
        events = np.asarray(arr[int(left):int(right), :4])
        proposer.propose(events)
        proposal_times.append((time.perf_counter() - t0) * 1000.0)
        total_events += int(right - left)
        calls += 1

    total_ms = (time.perf_counter() - total0) * 1000.0
    times = np.asarray(proposal_times, dtype=np.float64)
    print(f"mode={mode}")
    print(f"np_load_ms={load_ms:.3f}")
    print(f"calls={calls}")
    print(f"timed_total_ms={total_ms:.3f}")
    print(f"amortized_ms_per_window={total_ms/max(calls,1):.4f}")
    print(f"proposal_p50_ms={np.percentile(times,50):.4f}" if len(times) else "proposal_p50_ms=nan")
    print(f"proposal_p95_ms={np.percentile(times,95):.4f}" if len(times) else "proposal_p95_ms=nan")
    print(f"proposal_p99_ms={np.percentile(times,99):.4f}" if len(times) else "proposal_p99_ms=nan")
    print(f"mean_events_per_window={total_events/max(calls,1):.1f}")
    print("scope=timestamp-cache+vectorized-boundaries+sequential-slices+raw-proposal; np.load reported separately")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an offline/batched sequential OrbitSight proposal path")
    parser.add_argument("--npy", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--windows", type=int, default=500, help="0 means all available windows")
    parser.add_argument("--mode", choices=("memmap", "ram"), default="memmap")
    args = parser.parse_args()
    run(Path(args.npy), args.sequence, args.top_k, args.windows, args.mode)


if __name__ == "__main__":
    main()
