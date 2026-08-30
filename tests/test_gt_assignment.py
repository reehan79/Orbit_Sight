from orbitsight.io import Detection
from orbitsight.evaluation.gt_assignment import nearest_compatible_gt
from orbitsight.proposals import Candidate


def test_nearest_compatible_gt_picks_closer_of_two():
    candidate = Candidate(cx=10.0, cy=10.0, score=1.0, count=5, grid_x=1, grid_y=1)
    near = Detection(0, 40_000, 11.0, 10.0, 20.0, 20.0, 1.0)
    far = Detection(0, 40_000, 18.0, 10.0, 20.0, 20.0, 1.0)
    chosen = nearest_compatible_gt(candidate, [far, near], margin=4.0)
    assert chosen is near


def test_nearest_compatible_gt_none_when_incompatible():
    candidate = Candidate(cx=0.0, cy=0.0, score=1.0, count=1, grid_x=0, grid_y=0)
    gt = Detection(0, 40_000, 100.0, 100.0, 4.0, 4.0, 1.0)
    assert nearest_compatible_gt(candidate, [gt], margin=1.0) is None
