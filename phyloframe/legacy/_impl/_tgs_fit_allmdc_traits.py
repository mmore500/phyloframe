import typing

import numpy as np
import pandas as pd

from ..._auxlib._tgs_fit import TgsFitResult
from ._TgsAllmdcBranches import TgsAllmdcBranches
from ._TgsAllmdcPartitioning import TgsAllmdcPartitioning
from ._tgs_calc_allmdc_data import tgs_calc_allmdc_data
from ._tgs_calc_gain_ratios import tgs_calc_gain_ratios
from ._tgs_fit_allmdc_trait import tgs_fit_allmdc_trait


def tgs_fit_allmdc_traits(
    *,
    branches: TgsAllmdcBranches,
    partitioning: TgsAllmdcPartitioning,
    unit_weights: np.ndarray,
    neutral_traits: typing.Sequence[int],
    calibrate: bool,
    loss_rate: float,
    selection_starts: typing.Sequence[float],
    max_growth_exponent: float,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Fits each trait's gain rate and selection to branches, each weighted
    by `unit_weights` (e.g., bootstrap draw counts); see
    `tgs_calc_allmdc_nll` for the model. Returns one row per trait, with
    columns of `TgsFitResult` plus `gain_ratio`, the gain rate relative to
    neutral traits.
    """
    data = tgs_calc_allmdc_data(
        branches=branches,
        partitioning=partitioning,
        unit_weights=unit_weights,
        neutral_traits=neutral_traits,
        calibrate=calibrate,
    )
    # without gains, skip fitting (and its warning), as callers report such
    # traits and replicates
    fits = [
        tgs_fit_allmdc_trait(
            data=trait_data,
            survival_bin_times=partitioning.survival_bin_times,
            loss_rate=loss_rate,
            selection_starts=selection_starts,
            max_growth_exponent=max_growth_exponent,
        )
        if trait_data[1].any()
        else TgsFitResult(
            gain_rate=0.0, selection=np.nan, selection_lrt_naive=np.nan
        )
        for trait_data in data
    ]
    res = pd.DataFrame(fits, columns=TgsFitResult._fields)
    res["gain_ratio"] = tgs_calc_gain_ratios(
        gain_rates=res["gain_rate"].to_numpy(),
        neutral_traits=neutral_traits,
    )
    return res
