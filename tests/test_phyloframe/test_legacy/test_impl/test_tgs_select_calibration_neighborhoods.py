import logging

import numpy as np
import polars as pl

from phyloframe._auxlib._calc_group_means import calc_group_means
from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars
from phyloframe.legacy._impl import (
    tgs_count_traits,
    tgs_select_calibration_neighborhoods,
)

from ._simulate_allmdc_exposure_df import simulate_allmdc_exposure_df


def test_select(caplog):
    exposure_df = simulate_allmdc_exposure_df(8_000, num_neutral=3)
    exposure_df = exposure_df[exposure_df["clade_size"] == 1]
    coordinates = [
        exposure_df[c].to_numpy() for c in ("branch_length", "origin_time")
    ]
    bins = alifestd_mark_joint_quantile_bins_polars(
        pl.from_pandas(exposure_df[["branch_length", "origin_time"]]),
        bin_by=["branch_length", "origin_time"],
        max_bins=64,
    )["joint_quantile_bin"].to_numpy()
    num_bins = bins.max() + 1
    with caplog.at_level(logging.INFO):
        res = tgs_select_calibration_neighborhoods(
            bin_coords=calc_group_means(
                bins, np.column_stack(coordinates), num_groups=num_bins
            ),
            neighborhood_fractions=(0.25, 1.0),
            counts=tgs_count_traits(
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
