import numpy as np
import polars as pl

from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars
from phyloframe.legacy._impl import tgs_select_calibration_neighborhoods

from ._simulate_exposure_df import simulate_exposure_df


def _select(exposure_df: pl.DataFrame):
    return tgs_select_calibration_neighborhoods(
        exposure_df.with_columns(
            calibration_compare_by_branch_length=pl.col("episode_length"),
        ),
        neighborhood_fractions=(0.25, 1.0),
        neutral_trait_ids=[1, 2, 3],
        max_rate_ratio=np.inf,
    )


def test_block_diagonal():
    exposure_df = simulate_exposure_df(4_000, num_neutral=3).with_columns(
        partition_id=(pl.col("clade_size") > 1).cast(pl.Int64),
    )
    # bins numbered partition-major
    exposure_df = alifestd_mark_joint_quantile_bins_polars(
        exposure_df,
        bin_by=["episode_length", "origin_time"],
        partition_by="partition_id",
        max_num_bins=8,
        mark_as="bin_id",
    )
    neighborhoods = _select(exposure_df)
    num_bins = exposure_df["bin_id"].max() + 1
    assert neighborhoods.shape == (num_bins, num_bins)
    assert (neighborhoods.diagonal() == 1).all()
    # partition 0's bins come first
    first = exposure_df.filter(pl.col("partition_id") == 1)["bin_id"].min()
    dense = neighborhoods.toarray()
    assert not dense[:first, first:].any()
    assert not dense[first:, :first].any()


def test_one_bin_is_uncalibrated():
    neighborhoods = _select(simulate_exposure_df(400, num_neutral=3))
    assert neighborhoods.toarray().tolist() == [[1.0]]
