import numpy as np

from phyloframe.legacy._impl import tgs_find_allmdc_branches


def test_find():
    #         0
    #      1     6
    #    2   5
    #   3 4
    ancestor_ids = np.array([0, 0, 1, 2, 2, 1, 0])
    origin_time = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 3.0, 2.0])
    branch_lengths = origin_time - origin_time[ancestor_ids]
    branch_lengths[0] = np.nan
    trait_masks = [
        np.array([0, 0, 0, 1, 1, 0, 0], dtype=bool),  # gained on branch 2
        np.array([0, 0, 0, 1, 0, 1, 1], dtype=bool),  # leaves 3, 5, 6
    ]
    res = tgs_find_allmdc_branches(
        ancestor_ids=ancestor_ids,
        origin_time=origin_time,
        branch_lengths=branch_lengths,
        trait_masks=trait_masks,
    )
    # informative branches are those of nodes 1 to 6
    assert res.clade_sizes.tolist() == [3, 2, 1, 1, 1, 1]
    assert res.branch_lengths.tolist() == [1.0, 1.0, 2.0, 3.0, 2.0, 2.0]
    # to the latest leaf below: node 4 (time 5) for nodes 1 and 2
    assert res.survival_times.tolist() == [4.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    assert res.origin_times.tolist() == [1.0, 2.0, 4.0, 5.0, 3.0, 2.0]
    # trait 0: leaves 3 and 4 started with it, gained on branch 2
    assert res.trait_units[0].tolist() == [0, 1, 4, 5]
    assert res.trait_gains[0].tolist() == [1]
    assert res.trait_units[1].tolist() == [0, 1, 2, 3, 4, 5]
    assert res.trait_gains[1].tolist() == [2, 4, 5]


def test_zero_length_excluded():
    ancestor_ids = np.array([0, 0, 0])
    origin_time = np.array([0.0, 0.0, 1.0])
    res = tgs_find_allmdc_branches(
        ancestor_ids=ancestor_ids,
        origin_time=origin_time,
        branch_lengths=np.array([np.nan, 0.0, 1.0]),
        trait_masks=[np.array([0, 0, 1], dtype=bool)],
    )
    assert res.branch_lengths.tolist() == [1.0]
    assert res.trait_gains[0].tolist() == [0]
