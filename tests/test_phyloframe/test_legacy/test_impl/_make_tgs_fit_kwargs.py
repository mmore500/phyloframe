import typing

import numpy as np
import pandas as pd
from scipy import sparse

from phyloframe.legacy._alifestd_mark_joint_quantile_bins_polars import (
    _alifestd_mark_joint_quantile_bins_fast_path,
)


def make_tgs_fit_kwargs(
    exposure_df: pd.DataFrame, *, calibrate: bool = False, **kwargs
) -> typing.Dict[str, typing.Any]:
    """Arguments for `tgs_fit_traits`, but `weights`, treating traits after
    the first as neutral. If `calibrate`, bins episodes into 64
    equal-occupancy bins by branch length and origin time, each its own
    calibration neighborhood; else, one bin.
    """
    num_traits = sum(c.endswith("_starts_absent") for c in exposure_df)
    if calibrate:
        bins, num_bins = _alifestd_mark_joint_quantile_bins_fast_path(
            [
                exposure_df["branch_length"].to_numpy(),
                exposure_df["origin_time"].to_numpy(),
            ],
            max_bins=64,
            min_bin_size=4,
        )
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
