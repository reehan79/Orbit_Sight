from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import random


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[str, ...]
    validation: tuple[str, ...]


def infer_sensor(sequence: str) -> str:
    upper = sequence.upper()
    if upper.startswith("DAVIS"):
        return "DAVIS"
    if upper.startswith("DVX"):
        return "DVX"
    if "EVK4" in upper:
        return "EVK4"
    return "UNKNOWN"


def build_sequence_folds(sequences: list[str] | tuple[str, ...], n_splits: int = 5, seed: int = 20260825) -> list[Fold]:
    """Deterministic sequence-level folds, stratified by sensor where possible."""
    unique = sorted(set(sequences))
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(unique) < n_splits:
        raise ValueError("n_splits cannot exceed number of sequences")
    groups: dict[str, list[str]] = defaultdict(list)
    for sequence in unique:
        groups[infer_sensor(sequence)].append(sequence)
    rng = random.Random(seed)
    buckets: list[list[str]] = [[] for _ in range(n_splits)]
    offset = 0
    for sensor in sorted(groups):
        items = groups[sensor][:]
        rng.shuffle(items)
        for j, item in enumerate(items):
            buckets[(offset + j) % n_splits].append(item)
        offset = (offset + len(items)) % n_splits
    for empty_idx in [i for i, bucket in enumerate(buckets) if not bucket]:
        donor_idx = max(range(n_splits), key=lambda i: len(buckets[i]))
        if len(buckets[donor_idx]) <= 1:
            raise RuntimeError("Unable to construct non-empty validation folds")
        buckets[empty_idx].append(buckets[donor_idx].pop())
    folds: list[Fold] = []
    all_set = set(unique)
    for idx, validation in enumerate(buckets):
        val = tuple(sorted(validation))
        train = tuple(sorted(all_set - set(val)))
        folds.append(Fold(index=idx, train=train, validation=val))
    return folds
