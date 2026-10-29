import numpy as np

from phyloframe._auxlib._calc_group_means import calc_group_means


def test_1d():
    res = calc_group_means(
        np.array([0, 0, 2, 2, 2]),
        np.array([1.0, 3.0, 1.0, 2.0, 3.0]),
        num_groups=3,
    )
    np.testing.assert_array_equal(res, [2.0, np.nan, 2.0])


def test_2d():
    values = np.random.default_rng(1).random((100, 3))
    groups = np.random.default_rng(2).integers(0, 4, 100)
    res = calc_group_means(groups, values, num_groups=4)
    assert res.shape == (4, 3)
    for group in range(4):
        np.testing.assert_allclose(
            res[group], values[groups == group].mean(axis=0)
        )


def test_empty():
    res = calc_group_means(
        np.zeros(0, dtype=int), np.zeros((0, 2)), num_groups=2
    )
    assert res.shape == (2, 2)
    assert np.isnan(res).all()
