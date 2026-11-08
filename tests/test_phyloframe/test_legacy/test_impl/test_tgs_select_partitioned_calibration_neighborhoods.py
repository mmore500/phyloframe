import numpy as np
import polars as pl

from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars
from phyloframe.legacy._impl import (
    tgs_assign_calibration_bins,
    tgs_select_partitioned_calibration_neighborhoods,
)

from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df


def _select(exposure_df: pl.DataFrame):
    bins_df = tgs_assign_calibration_bins(exposure_df)
    return (
        bins_df,
        *tgs_select_partitioned_calibration_neighborhoods(
            exposure_df,
            bins_df=bins_df,
            num_bins=bins_df["bin"].max() + 1,
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
    bins_df, neighborhoods, trait_bin_counts_df = _select(exposure_df)
    num_bins = bins_df["bin"].max() + 1
    assert neighborhoods.shape == (num_bins, num_bins)
    assert (neighborhoods.diagonal() == 1).all()
    # partition 0's bins come first
    first = bins_df.filter(pl.col("partition") == 1)["bin"].min()
    dense = neighborhoods.toarray()
    assert not dense[:first, first:].any()
    assert not dense[first:, :first].any()
    assert trait_bin_counts_df["bin"].n_unique() == num_bins


def test_one_bin_is_uncalibrated():
    exposure_df = simulate_allmdc_exposure_df(400)
    __, neighborhoods, trait_bin_counts_df = _select(exposure_df)
    assert neighborhoods.toarray().tolist() == [[1.0]]
    assert trait_bin_counts_df["bin"].unique().to_list() == [0]


def test_bin_coords_ignore_row_order():
    from phyloframe.legacy._impl._tgs_select_partitioned_calibration_neighborhoods import (  # noqa: E501
        _tgs_calc_calibration_bin_coords,
    )

    rng = np.random.default_rng(1)
    exposure_df = pl.DataFrame(
        {
            "calibration_compare_by_x": rng.integers(0, 3, 200),  # ties
            "calibration_compare_by_c": np.ones(200),  # constant
        },
    )
    bins = rng.integers(0, 8, 200)
    expected = _tgs_calc_calibration_bin_coords(
        exposure_df, bins=bins, num_bins=8
    )
    order = rng.permutation(200)
    res = _tgs_calc_calibration_bin_coords(
        exposure_df[order], bins=bins[order], num_bins=8
    )
    np.testing.assert_allclose(res, expected)
    # the constant column places every bin alike
    assert np.ptp(res[:, 1]) == 0
