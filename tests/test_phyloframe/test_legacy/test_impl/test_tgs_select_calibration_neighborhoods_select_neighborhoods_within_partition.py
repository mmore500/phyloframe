import logging

import numpy as np
import polars as pl
import pytest

from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars
from phyloframe.legacy._impl import tgs_count_trait_bin_exposures
from phyloframe.legacy._impl._tgs_select_calibration_neighborhoods import (
    _tgs_select_neighborhoods_within_partition,
)

from ._simulate_exposure_df import simulate_exposure_df


@pytest.mark.parametrize("first_bin", [0, 20])
def test_select(first_bin: int, caplog):
    # the partition holds bins from first_bin on
    exposure_df = simulate_exposure_df(8_000, num_neutral=3)
    exposure_df = exposure_df.filter(pl.col("clade_size") == 1)
    exposure_df = alifestd_mark_joint_quantile_bins_polars(
        exposure_df,
        bin_by=["episode_length", "origin_time"],
        max_num_bins=64,
        mark_as="bin_id",
    )
    num_bins = exposure_df["bin_id"].max() + 1
    with caplog.at_level(logging.INFO):
        res = _tgs_select_neighborhoods_within_partition(
            exposure_df.select("episode_length", "origin_time", "bin_id")
            .group_by("bin_id")
            .mean()
            .sort("bin_id")
            .drop("bin_id")
            .to_numpy()[first_bin:],
            bin_ids=np.arange(first_bin, num_bins),
            num_bins=num_bins,
            neighborhood_fractions=(0.25, 1.0),
            trait_bin_counts_df=tgs_count_trait_bin_exposures(
                exposure_df,
                trait_ids=[1, 2, 3],
                weights=np.ones(len(exposure_df)),
            ),
            neutral_trait_ids=[1, 2, 3],
        )
    assert res.shape == (num_bins, num_bins)
    # each of the partition's bins is in its own neighborhood
    assert (res.diagonal()[first_bin:] == 1).all()
    # other bins have no neighborhood, and are in none
    dense = res.toarray()
    assert not dense[:first_bin].any()
    assert not dense[:, :first_bin].any()
    assert "selected calibration neighborhoods" in caplog.text
