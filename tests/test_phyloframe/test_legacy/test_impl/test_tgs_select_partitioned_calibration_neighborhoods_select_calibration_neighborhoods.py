import logging

import numpy as np
import polars as pl
import pytest

from phyloframe._auxlib._calc_group_means import calc_group_means
from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars
from phyloframe.legacy._impl import tgs_count_trait_bin_exposures
from phyloframe.legacy._impl._tgs_select_partitioned_calibration_neighborhoods import (
    _tgs_select_calibration_neighborhoods,
)

from ._simulate_exposure_df import simulate_exposure_df


def test_select(caplog):
    exposure_df = simulate_exposure_df(8_000, num_neutral=3)
    exposure_df = exposure_df.filter(pl.col("clade_size") == 1)
    coordinates = [
        exposure_df[c].to_numpy() for c in ("branch_length", "origin_time")
    ]
    bins = alifestd_mark_joint_quantile_bins_polars(
        exposure_df,
        bin_by=["branch_length", "origin_time"],
        max_num_bins=64,
    )["joint_quantile_bin"].to_numpy()
    num_bins = bins.max() + 1
    with caplog.at_level(logging.INFO):
        res = _tgs_select_calibration_neighborhoods(
            bin_coords=calc_group_means(
                bins, np.column_stack(coordinates), num_groups=num_bins
            ),
            neighborhood_fractions=(0.25, 1.0),
            trait_bin_counts_df=tgs_count_trait_bin_exposures(
                exposure_df=exposure_df,
                bins=bins,
                num_bins=num_bins,
                weights=np.ones(len(exposure_df)),
            ),
            neutral_traits=[1, 2, 3],
            trait_names=["focal", "a", "b", "c"],
            max_rate_ratio=np.inf,
        )
    assert res.shape == (num_bins, num_bins)
    # each bin is in its own neighborhood
    assert (res.diagonal() == 1).all()
    assert "selected calibration neighborhoods" in caplog.text


def _select_two_bins(neighborhood_fractions):
    # neutral trait 2 gains only in bin 0, so cannot calibrate neutral trait
    # 1 in bin 1 with neighborhoods of that bin alone
    num_successes = np.array([[5, 5], [10, 10], [10, 0]], dtype=float)
    return _tgs_select_calibration_neighborhoods(
        bin_coords=np.array([[0.0], [1.0]]),
        neighborhood_fractions=neighborhood_fractions,
        trait_bin_counts_df=pl.DataFrame(
            {
                "trait_id": np.repeat(np.arange(3), 2),
                "bin": np.tile(np.arange(2), 3),
                "num_trials": 100.0,
                "num_successes": num_successes.ravel(),
                "length_sum": 1000.0,
            },
        ),
        neutral_traits=[1, 2],
        trait_names=["focal", "a", "b"],
        max_rate_ratio=np.inf,
    )


def test_skips_neighborhoods_without_reference_gains():
    res = _select_two_bins((0.5, 1.0))
    assert (res.toarray() == 1).all()


def test_too_fine_raises():
    with pytest.raises(ValueError, match="coarser calibration"):
        _select_two_bins((0.5,))
