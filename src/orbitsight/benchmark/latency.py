from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Iterable
import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    calls: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def benchmark(fn: Callable[[object], object], items: Iterable[object], warmup: int = 5) -> LatencyStats:
    items = list(items)
    if not items:
        raise ValueError("No benchmark items supplied")
    for item in items[: min(warmup, len(items))]:
        fn(item)
    samples: list[float] = []
    for item in items:
        start = perf_counter_ns()
        fn(item)
        samples.append((perf_counter_ns() - start) / 1_000_000.0)
    a = np.asarray(samples, dtype=np.float64)
    return LatencyStats(len(samples), float(np.mean(a)), float(np.percentile(a, 50)), float(np.percentile(a, 95)), float(np.percentile(a, 99)))
