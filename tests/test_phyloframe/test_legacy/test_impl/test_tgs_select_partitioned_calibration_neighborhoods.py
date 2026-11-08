import numpy as np
import polars as pl

from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars
from phyloframe.legacy._impl import (
    tgs_assign_calibration_bins,
    tgs_select_partitioned_calibration_neighborhoods,
)

from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df


def _select(exposure_df: pl.DataFrame):
    bins, partition_bounds = tgs_assign_calibration_bins(exposure_df)
    return (
        partition_bounds,
        *tgs_select_partitioned_calibration_neighborhoods(
            exposure_df,
            bins=bins,
            partition_bounds=partition_bounds,
            neighborhood_fractions=(0.25, 1.0),
            neutral_traits=[1, 2, 3],
            trait_names=["focal", "a", "b", "c"],
            max_rate_ratio=np.inf,
        ),
    )


def test_block_diagonal():
    exposure_df = simulate_allmdc_exposure_df(4_000, num_neutral=3)
    exposure_df = alifestd_mark_joint_quantile_bins_polars(
        exposure_df,
        bin_by=["branch_length", "origin_time"],
        max_num_bins=8,
        mark_as="calibration_bin",
    ).with_columns(
        calibration_partition=(pl.col("clade_size") > 1).cast(pl.Int64),
        calibration_compare_by_branch_length=pl.col("branch_length"),
    )
    partition_bounds, neighborhoods, counts_df = _select(exposure_df)
    num_bins = partition_bounds[-1]
    assert neighborhoods.shape == (num_bins, num_bins)
    assert (neighborhoods.diagonal() == 1).all()
    first = partition_bounds[1]
    dense = neighborhoods.toarray()
    assert not dense[:first, first:].any()
    assert not dense[first:, :first].any()
    assert counts_df is not None


def test_one_bin_is_uncalibrated():
    exposure_df = simulate_allmdc_exposure_df(400)
    __, neighborhoods, counts_df = _select(exposure_df)
    assert neighborhoods.toarray().tolist() == [[1.0]]
    assert counts_df is None
