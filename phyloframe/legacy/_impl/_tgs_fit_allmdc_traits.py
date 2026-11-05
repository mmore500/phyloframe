import typing

import numpy as np
import pandas as pd

from ._tgs_calc_allmdc_data import tgs_calc_allmdc_data
from ._tgs_fit_trait_partitions import tgs_fit_trait_partitions


def tgs_fit_allmdc_traits(
    *,
    exposure_df: pd.DataFrame,
    partitioning: typing.Dict[str, typing.Any],
    weights: np.ndarray,
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    growth_exponent_bound: float,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits each trait to all branches, each weighted by `weights` (e.g.,
    bootstrap draw counts), returning one row per trait (see
    `tgs_fit_trait_partitions`); see `tgs_calc_nll` for the model.
    """
    data, survival_shares = tgs_calc_allmdc_data(
        exposure_df=exposure_df,
        partitioning=partitioning,
        weights=weights,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
    )
    return tgs_fit_trait_partitions(
        data=data,
        neutral_traits=neutral_traits,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
        growth_exponent_bound=growth_exponent_bound,
        survival_shares=survival_shares,
        distinct_survival_times=partitioning["distinct_survival_times"],
    )
