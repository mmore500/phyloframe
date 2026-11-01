import typing

import numpy as np
import pandas as pd

from ._tgs_calc_allmdc_data import tgs_calc_allmdc_data
from ._tgs_calc_gain_ratios import tgs_calc_gain_ratios
from ._tgs_fit_allmdc_trait import tgs_fit_allmdc_trait


def tgs_fit_allmdc_traits(
    *,
    branches: pd.DataFrame,
    partitioning: typing.Dict[str, typing.Any],
    unit_weights: np.ndarray,
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    growth_exponent_bound: float,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits each trait's gain rate and selection to branches, each weighted
    by `unit_weights` (e.g., bootstrap draw counts); see
    `tgs_calc_allmdc_nll` for the model. Returns one row per trait, with
    columns of `TgsFitResult` (see `tgs_fit_allmdc_trait`) plus
    `gain_ratio`, the gain rate relative to neutral traits.
    """
    data, survival_shares = tgs_calc_allmdc_data(
        branches=branches,
        partitioning=partitioning,
        unit_weights=unit_weights,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
    )
    res = pd.DataFrame(
        [
            tgs_fit_allmdc_trait(
                partitions=partitions,
                survival_shares=survival_shares[partitions.index],
                survival_bin_times=partitioning["survival_bin_times"],
                loss_rate=loss_rate,
                selection_starts=selection_starts,
                growth_exponent_bound=growth_exponent_bound,
            )
            for __, partitions in data.groupby("trait", observed=False)
        ],
    )
    res["gain_ratio"] = tgs_calc_gain_ratios(
        gain_rates=res["gain_rate"].to_numpy(),
        neutral_traits=neutral_traits,
    )
    return res
