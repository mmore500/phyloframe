import numpy as np
import pytest

from phyloframe.legacy._impl import tgs_calc_trait_states

#     0
#   1   2
#   3
_ANCESTOR_IDS = np.array([0, 0, 0, 1])


@pytest.mark.parametrize(
    "transitions, expected",
    [
        ([0, 1, 0, 0], [0, 1, 0, 1]),
        ([1, 0, 0, 0], [1, 1, 1, 1]),
        ([0, 0, 1, 0], [0, 0, 1, 0]),
        (np.array([0, 1, 0, 0], dtype=np.int8), [0, 1, 0, 1]),
        (np.array([False, True, False, False]), [0, 1, 0, 1]),
    ],
)
def test_states(transitions, expected):
    res = tgs_calc_trait_states(
        ancestor_ids=_ANCESTOR_IDS,
        transitions=np.asarray(transitions),
        trait_name="t",
    )
    assert res.tolist() == expected


@pytest.mark.parametrize(
    "transitions, error",
    [
        (np.array([0.0, 1.0, 0.0, 0.0]), NotImplementedError),
        (np.array([0, 2, 0, 0]), ValueError),
        (np.array([0, 1, 0, 1]), ValueError),  # state 2 at node 3
        (np.array([1, -1, 0, 0]), NotImplementedError),  # loss
    ],
)
def test_invalid(transitions, error):
    with pytest.raises(error):
        tgs_calc_trait_states(
            ancestor_ids=_ANCESTOR_IDS, transitions=transitions, trait_name="t"
        )
