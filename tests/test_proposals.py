import numpy as np
from orbitsight.proposals.raw_candidates import RawGridProposer


def test_strongest_cell_ranks_first():
    proposer = RawGridProposer(width=32, height=32, cell_size=8, top_k=2)
    a = np.array([[9, 9]] * 6 + [[25, 25]] * 2, dtype=float)
    candidates = proposer.propose(a)
    assert len(candidates) == 2
    assert (candidates[0].grid_x, candidates[0].grid_y) == (1, 1)
    assert candidates[0].count == 6


def test_topk_tie_breaks_by_lowest_cell_id():
    proposer = RawGridProposer(width=32, height=32, cell_size=8, top_k=1)
    events = np.array([[1, 1]] * 3 + [[9, 9]] * 3, dtype=float)
    first = proposer.propose(events)
    second = proposer.propose(events)
    assert len(first) == 1
    assert (first[0].grid_x, first[0].grid_y) == (0, 0)
    assert first[0].count == 3
    assert [(c.grid_x, c.grid_y, c.count) for c in first] == [(c.grid_x, c.grid_y, c.count) for c in second]


def test_topk_boundary_tie_prefers_lower_cell_id():
    proposer = RawGridProposer(width=48, height=32, cell_size=8, top_k=2)
    events = np.array(
        [[1, 1]] * 4
        + [[9, 9]] * 4
        + [[17, 1]] * 2
        + [[25, 1]] * 2,
        dtype=float,
    )
    candidates = proposer.propose(events)
    assert len(candidates) == 2
    assert candidates[0].count == 4
    assert candidates[1].count == 4
    assert (candidates[0].grid_x, candidates[0].grid_y) <= (candidates[1].grid_x, candidates[1].grid_y)

