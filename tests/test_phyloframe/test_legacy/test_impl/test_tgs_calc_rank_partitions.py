import numpy as np

from phyloframe.legacy._impl import tgs_calc_rank_partitions


def test_partition():
    rng = np.random.default_rng(1)
    x, y = rng.random(1000), rng.random(1000)
    partitions, num_partitions, coords = tgs_calc_rank_partitions(
        coordinates=[x, y], max_partitions=16
    )
    assert num_partitions == 16
    assert set(np.bincount(partitions, minlength=16)) <= {62, 63}
    assert coords.shape == (16, 2)
    assert ((coords > 0) & (coords < 1)).all()


def test_keeps_four_per_partition():
    partitions, num_partitions, __ = tgs_calc_rank_partitions(
        coordinates=[np.arange(10.0)], max_partitions=1024
    )
    assert num_partitions == 2
    assert np.bincount(partitions).tolist() == [5, 5]


def test_ties_broken_by_other_coordinates():
    rng = np.random.default_rng(2)
    tied = np.zeros(200)
    other = rng.random(200)
    order = rng.permutation(200)
    partitions, __, __ = tgs_calc_rank_partitions(
        coordinates=[tied, other], max_partitions=8
    )
    shuffled, __, __ = tgs_calc_rank_partitions(
        coordinates=[tied[order], other[order]], max_partitions=8
    )
    assert (shuffled == partitions[order]).all()
