import typing

import numpy as np
import polars as pl
from scipy import sparse

from phyloframe.legacy import alifestd_mark_joint_quantile_bins_polars


def make_tgs_fit_kwargs(
    exposure_df: pl.DataFrame, *, calibrate: bool = False, **kwargs
) -> typing.Dict[str, typing.Any]:
    """Arguments for `tgs_fit_traits`, but `weights`, treating traits after
    the first as neutral. If `calibrate`, bins episodes into 64
    equal-occupancy bins by branch length and origin time, each its own
    calibration neighborhood; else, one bin.
    """
    num_traits = sum(c.endswith("_starts_absent") for c in exposure_df.columns)
    if calibrate:
        bins = alifestd_mark_joint_quantile_bins_polars(
            exposure_df,
            bin_by=["branch_length", "origin_time"],
            max_bins=64,
        )["joint_quantile_bin"].to_numpy()
        num_bins = bins.max() + 1
    else:
        bins, num_bins = np.zeros(len(exposure_df), dtype=np.int64), 1
    distinct_survival_durations, survival_duration_indices = np.unique(
        exposure_df["survival_duration"].to_numpy(), return_inverse=True
    )
    return {
        **dict(
            exposure_df=exposure_df,
            bins=bins,
            num_bins=num_bins,
            calibration_neighborhoods=sparse.identity(num_bins, format="csr"),
            distinct_survival_durations=distinct_survival_durations,
            survival_duration_indices=survival_duration_indices,
            neutral_traits=list(range(1, num_traits)),
            calibrate_max_rate_ratio=np.inf,
            length_rtol=1e-3,
            loss_rate=0.0,
            selection_coef_starts=(0.0, -0.3, -1.8),
            growth_exponent_bound=600.0,
        ),
        **kwargs,
    }
