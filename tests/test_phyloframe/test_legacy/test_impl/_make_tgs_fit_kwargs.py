import typing

import polars as pl
from scipy import sparse

from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars


def make_tgs_fit_kwargs(
    exposure_df: pl.DataFrame, *, calibrate: bool = False, **kwargs
) -> typing.Dict[str, typing.Any]:
    """Arguments for `tgs_fit_traits`, but `weights` and `progress_wrap`,
    treating traits after the first as neutral. If `calibrate`, bins
    episodes into 64 equal-occupancy bins by episode duration and origin time,
    each its own calibration neighborhood; else, one bin.
    """
    num_traits = sum(
        c.endswith("_episode_starts_absent") for c in exposure_df.columns
    )
    if calibrate:
        exposure_df = alifestd_mark_joint_quantile_bins_polars(
            exposure_df,
            bin_by=["episode_duration", "origin_time"],
            max_num_bins=64,
            mark_as="bin_id",
        )
    num_bins = exposure_df["bin_id"].max() + 1
    return {
        **dict(
            exposure_df=exposure_df,
            num_traits=num_traits,
            calibration_neighborhoods=sparse.identity(num_bins, format="csr"),
            neutral_trait_ids=list(range(1, num_traits)),
            exposure_duration_rtol=1e-3,
            loss_rate=0.0,
            fit_selection_coef_starts=(0.0, -0.3, -1.8),
            fit_growth_exponent_bound=600.0,
        ),
        **kwargs,
    }
