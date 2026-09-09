from __future__ import annotations

import math

from orbitsight.io import Detection
from orbitsight.proposals import Candidate


def compatible(candidate: Candidate, gt: Detection, margin: float) -> bool:
    return (
        abs(candidate.cx - gt.cx) <= gt.width / 2.0 + margin
        and abs(candidate.cy - gt.cy) <= gt.height / 2.0 + margin
    )


def nearest_compatible_gt(
    candidate: Candidate,
    gts: list[Detection],
    margin: float,
) -> Detection | None:
    """Assign exactly one GT: nearest compatible by Euclidean centre distance."""
    matches = [gt for gt in gts if compatible(candidate, gt, margin)]
    if not matches:
        return None
    return min(matches, key=lambda gt: math.hypot(candidate.cx - gt.cx, candidate.cy - gt.cy))
