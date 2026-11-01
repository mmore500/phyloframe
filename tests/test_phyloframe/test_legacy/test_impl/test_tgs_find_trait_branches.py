import pandas as pd

from phyloframe.legacy._impl import tgs_find_trait_branches


def test_find():
    branches = pd.DataFrame(
        {
            "branch_length": [1.0, 2.0, 3.0],
            "trait0_starts_absent": [True, False, True],
            "trait0_gain": [False, False, True],
            "trait1_starts_absent": [True, True, True],
            "trait1_gain": [True, False, False],
        },
    )
    units, gains = tgs_find_trait_branches(branches)
    assert [u.tolist() for u in units] == [[0, 2], [0, 1, 2]]
    assert [g.tolist() for g in gains] == [[2], [0]]


def test_no_traits():
    units, gains = tgs_find_trait_branches(pd.DataFrame({"x": [1.0]}))
    assert units == [] and gains == []
