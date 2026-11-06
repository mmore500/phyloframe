import typing

import numpy as np
import pandas as pd

from ._tgs_allmdc_tabulate_partitions import tgs_allmdc_tabulate_partitions
from ._tgs_fit_trait_partitions import tgs_fit_trait_partitions


def tgs_allmdc_fit_traits(
    *,
    exposure_df: pd.DataFrame,
    partitioning: typing.Dict[str, typing.Any],
    weights: np.ndarray,
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    calibrate_max_rate_ratio: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_polars`. Acronym
    tgs stands for trait gain/selection.

    Fits each trait to all branches, each weighted by `weights` (e.g.,
    bootstrap draw counts), returning one row per trait (see
    `tgs_fit_trait_partitions`); see `tgs_calc_nll` for the model.
    """
    partition_df, survival_shares = tgs_allmdc_tabulate_partitions(
        exposure_df=exposure_df,
        partitioning=partitioning,
        weights=weights,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
    )
    return tgs_fit_trait_partitions(
        partition_df=partition_df,
        neutral_traits=neutral_traits,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
        survival_shares=survival_shares,
        distinct_survival_times=partitioning["distinct_survival_times"],
    )
