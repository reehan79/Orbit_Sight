import numpy as np
from orbitsight.proposals.raw_candidates import RawGridProposer


def test_strongest_cell_ranks_first():
    proposer = RawGridProposer(width=32, height=32, cell_size=8, top_k=2)
    a = np.array([[9, 9]] * 6 + [[25, 25]] * 2, dtype=float)
    candidates = proposer.propose(a)
    assert len(candidates) == 2
    assert (candidates[0].grid_x, candidates[0].grid_y) == (1, 1)
    assert candidates[0].count == 6
