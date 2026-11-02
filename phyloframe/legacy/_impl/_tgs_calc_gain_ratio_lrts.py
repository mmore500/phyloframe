import typing

import numpy as np
import pandas as pd

from ._tgs_find_reference_traits import tgs_find_reference_traits
from ._tgs_fit import _tgs_minimize_nll_fixed_gain


def tgs_calc_gain_ratio_lrts(
    *,
    fits: pd.DataFrame,
    calc_nlls: typing.Sequence[typing.Optional[typing.Callable]],
    selection_scales: typing.Sequence[float],
    neutral_traits: typing.Sequence[int],
    loss_rate: float,
    selection_starts: typing.Sequence[float],
) -> np.ndarray:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns, per trait, the likelihood-ratio test (LRT) statistic for gain
    ratio = 1, a profile likelihood test: twice the increase in negative
    log likelihood from holding the trait's gain rate at the mean fitted
    gain rate of its neutral reference (see `tgs_find_reference_traits`)
    and refitting selection alone. `fits` holds each trait's fitted
    `gain_rate` and `selection`; `calc_nlls`, each trait's `calc_nll(log
    gain rate, rate)` (None without gains); and `selection_scales`, their
    selection scales (see `_tgs_minimize_nll`).

    The statistic naively treats the reference gain rate as known, and
    branches (or bins) as independent, so overstates significance. It is
    nan wherever the gain ratio is undefined, or the trait has no gains.
    """
    gain_rates = fits["gain_rate"].to_numpy()
    selections = fits["selection"].to_numpy()
    lrts = np.full(len(fits), np.nan)
    for trait, calc_nll in enumerate(calc_nlls):
        refs = tgs_find_reference_traits(
            neutral_traits=neutral_traits, trait=trait
        )
        ref_gain_rate = gain_rates[refs].mean() if len(refs) else 0.0
        if calc_nll is None or not ref_gain_rate > 0:
            continue
        free_nll, __, __ = calc_nll(
            np.log(gain_rates[trait]), selections[trait] - loss_rate
        )
        null_nll = _tgs_minimize_nll_fixed_gain(
            calc_nll=calc_nll,
            log_gain_rate=np.log(ref_gain_rate),
            selection_scale=selection_scales[trait],
            loss_rate=loss_rate,
            selection_starts=[
                selections[trait] / selection_scales[trait],
                *selection_starts,
            ],
        )
        lrts[trait] = 2 * (null_nll - free_nll)
    return lrts
