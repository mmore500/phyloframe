import numpy as np

from phyloframe._auxlib import calc_quantile_bins


def test_equal_occupancy():
    values = np.random.default_rng(1).permutation(1_000)
    bins, num_bins = calc_quantile_bins(values, max_bins=8)
    assert num_bins == 8
    assert np.bincount(bins).tolist() == [125] * 8
    # ordered by value
    assert (np.diff(bins[np.argsort(values)]) >= 0).all()


def test_ties_merge_bins():
    values = np.r_[np.zeros(900), np.arange(1, 101)]
    bins, num_bins = calc_quantile_bins(values, max_bins=10)
    assert num_bins < 10
    assert bins.max() == num_bins - 1
    assert len(np.unique(bins[values == 0])) == 1


def test_constant():
    bins, num_bins = calc_quantile_bins(np.full(5, 3.0), max_bins=4)
    assert num_bins == 1
    assert (bins == 0).all()


def test_empty():
    bins, num_bins = calc_quantile_bins(np.array([]), max_bins=4)
    assert len(bins) == 0
    assert num_bins == 1
