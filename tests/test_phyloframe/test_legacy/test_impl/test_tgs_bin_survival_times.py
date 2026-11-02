import numpy as np

from phyloframe.legacy._impl import tgs_bin_survival_times


def test_equal_occupancy():
    times = np.random.default_rng(1).permutation(800).astype(float)
    bins, bin_times = tgs_bin_survival_times(times, max_bins=8)
    assert np.bincount(bins).tolist() == [100] * 8
    assert (np.diff(bin_times) > 0).all()
    for b, mean in enumerate(bin_times):
        assert times[bins == b].mean() == mean


def test_ties_merge_bins():
    times = np.r_[np.zeros(900), np.arange(1, 101.0)]
    bins, bin_times = tgs_bin_survival_times(times, max_bins=10)
    assert len(bin_times) == bins.max() + 1 < 10
    assert len(np.unique(bins[times == 0])) == 1
    assert bin_times[0] == 0.0


def test_constant():
    bins, bin_times = tgs_bin_survival_times(np.full(5, 3.0), max_bins=4)
    assert (bins == 0).all()
    assert bin_times.tolist() == [3.0]
