import numpy as np
import pytest

from phyloframe.legacy._alifestd_mark_joint_quantile_bins_polars import (
    _alifestd_mark_joint_quantile_bins_fast_path,
)


def test_bins():
    rng = np.random.default_rng(1)
    x, y = rng.random(1000), rng.random(1000)
    bins = _alifestd_mark_joint_quantile_bins_fast_path(
        [x, y], max_num_bins=16, min_bin_occupancy=4
    )
    assert set(np.bincount(bins, minlength=16)) <= {62, 63}


@pytest.mark.parametrize("max_num_bins", [1, 2, 5, 8, 1024])
@pytest.mark.parametrize("min_bin_occupancy", [1, 2, 4, 10, 20])
@pytest.mark.parametrize("num_items", [5, 9, 10, 100])
def test_bin_limits(max_num_bins: int, min_bin_occupancy: int, num_items: int):
    rng = np.random.default_rng(num_items)
    bins = _alifestd_mark_joint_quantile_bins_fast_path(
        [rng.random(num_items), rng.random(num_items)],
        max_num_bins=max_num_bins,
        min_bin_occupancy=min_bin_occupancy,
    )
    counts = np.bincount(bins)
    assert counts.sum() == num_items
    assert len(counts) <= max_num_bins
    assert counts.min() >= min(min_bin_occupancy, num_items)


def test_empty():
    bins = _alifestd_mark_joint_quantile_bins_fast_path(
        [np.zeros(0)], max_num_bins=8, min_bin_occupancy=4
    )
    assert len(bins) == 0


def test_ties_broken_by_other_coordinates():
    rng = np.random.default_rng(2)
    tied = np.zeros(200)
    other = rng.random(200)
    order = rng.permutation(200)
    bins = _alifestd_mark_joint_quantile_bins_fast_path(
        [tied, other], max_num_bins=8, min_bin_occupancy=4
    )
    shuffled = _alifestd_mark_joint_quantile_bins_fast_path(
        [tied[order], other[order]], max_num_bins=8, min_bin_occupancy=4
    )
    assert (shuffled == bins[order]).all()
