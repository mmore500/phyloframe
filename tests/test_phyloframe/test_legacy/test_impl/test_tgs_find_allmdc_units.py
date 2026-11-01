import numpy as np

from phyloframe.legacy._impl import tgs_find_allmdc_units


def test_groups():
    #       0
    #     1   4
    #    2 3
    units = tgs_find_allmdc_units(
        ancestor_ids=np.array([0, 0, 1, 1, 0]),
        branch_lengths=np.array([np.nan, 1.0, 2.0, 3.0, 4.0]),
        trait_masks=[np.array([0, 0, 1, 1, 0], dtype=bool)],
        groups_per_doubling=2,
    )
    assert units.group_bounds == [(1, 1), (2, 2)]
    assert units.group_nodes[0].tolist() == [2, 3, 4]
    assert units.group_nodes[1].tolist() == [1]
    # leaf 4 started without the trait; leaves 2, 3 started with it
    assert units.trait_units[0][0].tolist() == [2]
    assert units.trait_gains[0][0].tolist() == []
    assert units.trait_units[1][0].tolist() == [0]
    assert units.trait_gains[1][0].tolist() == [0]


def test_zero_length_excluded():
    units = tgs_find_allmdc_units(
        ancestor_ids=np.array([0, 0, 0]),
        branch_lengths=np.array([np.nan, 0.0, 1.0]),
        trait_masks=[np.array([0, 1, 1], dtype=bool)],
        groups_per_doubling=2,
    )
    assert units.group_nodes[0].tolist() == [2]
    assert units.group_bounds == [(1, 1)]
