import numpy as np

from phyloframe._auxlib import calc_tiebroken_ranks


def test_distinct():
    ranks = calc_tiebroken_ranks([np.array([3.0, 1.0, 2.0])])
    np.testing.assert_array_equal(ranks[:, 0], [3, 1, 2])


def test_ties_broken_by_other_coordinates():
    x = np.array([0.0, 0.0, 0.0, 1.0])
    y = np.array([3.0, 1.0, 2.0, 0.0])
    ranks = calc_tiebroken_ranks([x, y])
    assert ranks.shape == (4, 2)
    np.testing.assert_array_equal(ranks[:, 0], [3, 1, 2, 4])
    np.testing.assert_array_equal(ranks[:, 1], [4, 2, 3, 1])


def test_independent_of_item_order():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, 100).astype(float)
    y = rng.random(100)
    order = rng.permutation(100)
    np.testing.assert_array_equal(
        calc_tiebroken_ranks([x, y])[order],
        calc_tiebroken_ranks([x[order], y[order]]),
    )


def test_full_ties_broken_by_item_order():
    x = np.array([1.0, 0.0, 1.0])
    ranks = calc_tiebroken_ranks([x, x])
    np.testing.assert_array_equal(ranks, [[2, 2], [1, 1], [3, 3]])
