from .candidate_ranker import (
    FEATURE_COLUMNS,
    RankerBundle,
    fit_bbox_ridge,
    fit_rankers,
    score_ranker,
)

# TinyFoveatedRefiner requires torch; keep optional so submission/CPU images
# do not import torch at package load time.
try:
    from .foveated_refiner import TinyFoveatedRefiner, parameter_count
except ImportError:  # pragma: no cover - torch optional for deploy
    TinyFoveatedRefiner = None  # type: ignore[misc, assignment]
    parameter_count = None  # type: ignore[misc, assignment]

__all__ = [
    "FEATURE_COLUMNS",
    "RankerBundle",
    "TinyFoveatedRefiner",
    "fit_bbox_ridge",
    "fit_rankers",
    "parameter_count",
    "score_ranker",
]
