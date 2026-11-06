import numpy as np
import pytest

from phyloframe.legacy._alifestd_mark_joint_quantile_bins_polars import (
    _alifestd_mark_joint_quantile_bins_fast_path,
)


def test_bins():
    rng = np.random.default_rng(1)
    x, y = rng.random(1000), rng.random(1000)
    bins, num_bins = _alifestd_mark_joint_quantile_bins_fast_path(
        [x, y], max_bins=16, min_bin_size=4
    )
    assert num_bins == 16
    assert set(np.bincount(bins, minlength=16)) <= {62, 63}


@pytest.mark.parametrize("min_bin_size", [1, 2, 4, 10, 20])
def test_min_bin_size(min_bin_size: int):
    bins, num_bins = _alifestd_mark_joint_quantile_bins_fast_path(
        [np.arange(10.0)], max_bins=1024, min_bin_size=min_bin_size
    )
    assert num_bins == 2 ** int(max(np.log2(10 / min_bin_size), 0))
    assert np.bincount(bins).sum() == 10
    assert np.bincount(bins).min() >= min(min_bin_size, 10) // 2


def test_empty():
    bins, num_bins = _alifestd_mark_joint_quantile_bins_fast_path(
        [np.zeros(0)], max_bins=8, min_bin_size=4
    )
    assert len(bins) == 0
    assert num_bins == 0


def test_ties_broken_by_other_coordinates():
    rng = np.random.default_rng(2)
    tied = np.zeros(200)
    other = rng.random(200)
    order = rng.permutation(200)
    bins, __ = _alifestd_mark_joint_quantile_bins_fast_path(
        [tied, other], max_bins=8, min_bin_size=4
    )
    shuffled, __ = _alifestd_mark_joint_quantile_bins_fast_path(
        [tied[order], other[order]], max_bins=8, min_bin_size=4
    )
    assert (shuffled == bins[order]).all()
