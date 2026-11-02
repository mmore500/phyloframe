import typing

import numpy as np
import pandas as pd

from ._tgs_calc_allmdc_data import tgs_calc_allmdc_data
from ._tgs_calc_gain_ratio_lrts import tgs_calc_gain_ratio_lrts
from ._tgs_calc_gain_ratios import tgs_calc_gain_ratios
from ._tgs_fit_allmdc_trait import (
    tgs_fit_allmdc_trait,
    tgs_make_allmdc_trait_nll,
)


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

    Fits each trait's gain rate and selection to branches, each weighted
    by `weights` (e.g., bootstrap draw counts); see
    `tgs_calc_allmdc_nll` for the model. Returns one row per trait, with
    columns of `TgsFitResult` (see `tgs_fit_allmdc_trait`) plus
    `gain_ratio`, the gain rate relative to neutral traits, and
    `gain_ratio_lrt_naive` (see `tgs_calc_gain_ratio_lrts`).
    """
    data, survival_shares = tgs_calc_allmdc_data(
        exposure_df=exposure_df,
        partitioning=partitioning,
        weights=weights,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
    )
    fits, calc_nlls, selection_scales = [], [], []
    for __, partitions in data.groupby("trait", observed=False):
        calc_nll, selection_scale = tgs_make_allmdc_trait_nll(
            partitions=partitions,
            survival_shares=survival_shares[partitions.index],
            distinct_survival_times=partitioning["distinct_survival_times"],
            growth_exponent_bound=growth_exponent_bound,
        )
        fit = tgs_fit_allmdc_trait(
            calc_nll=calc_nll,
            selection_scale=selection_scale,
            partitions=partitions,
            loss_rate=loss_rate,
            selection_starts=selection_starts,
        )
        has_gains = partitions["num_successes"].to_numpy().any()
        fits.append(fit)
        calc_nlls.append(calc_nll if has_gains else None)
        selection_scales.append(selection_scale)
    res = pd.DataFrame(fits)
    res["gain_ratio"] = tgs_calc_gain_ratios(
        gain_rates=res["gain_rate"].to_numpy(),
        neutral_traits=neutral_traits,
    )
    res["gain_ratio_lrt_naive"] = tgs_calc_gain_ratio_lrts(
        fits=res,
        calc_nlls=calc_nlls,
        selection_scales=selection_scales,
        neutral_traits=neutral_traits,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
    )
    return res
