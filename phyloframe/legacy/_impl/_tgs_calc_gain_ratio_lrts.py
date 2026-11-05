import typing
import warnings

import numpy as np
import pandas as pd

from ._tgs_find_reference_traits import tgs_find_reference_traits
from ._tgs_fit import _tgs_calc_scales, tgs_minimize_nll


def tgs_calc_gain_ratio_lrts(
    *,
    fits: pd.DataFrame,
    trait_partitions: typing.Sequence[pd.DataFrame],
    calc_nlls: typing.Sequence[typing.Callable],
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
    `gain_rate` and `selection`; `trait_partitions`, each trait's
    partitions (see `tgs_fit_partitions`); and `calc_nlls`, their
    `calc_nll(log_gain_rate=..., rate=...)` (see `tgs_calc_nll`).

    The statistic naively treats the reference gain rate as known, and
    branches (or bins) as independent, so overstates significance. It is
    nan wherever the gain ratio is undefined, or the trait has no gains.
    """
    gain_rates = fits["gain_rate"].to_numpy()
    selections = fits["selection"].to_numpy()
    lrts = np.full(len(fits), np.nan)
    for trait, (partitions, calc_nll) in enumerate(
        zip(trait_partitions, calc_nlls),
    ):
        refs = tgs_find_reference_traits(
            neutral_traits=neutral_traits, trait=trait
        )
        ref_gain_rate = gain_rates[refs].mean() if len(refs) else 0.0
        if not partitions["num_successes"].to_numpy().any() or not (
            ref_gain_rate > 0
        ):
            continue
        selection_scale, __ = _tgs_calc_scales(partitions)
        free_nll, __, __ = calc_nll(
            log_gain_rate=np.log(gain_rates[trait]),
            rate=selections[trait] - loss_rate,
        )
        null = tgs_minimize_nll(
            calc_nll=calc_nll,
            # from the free fit, then from each of `selection_starts`
            starts=[
                (np.log(ref_gain_rate), start)
                for start in (
                    selections[trait] / selection_scale,
                    *selection_starts,
                )
            ],
            is_free=(False, True),
            selection_scale=selection_scale,
            loss_rate=loss_rate,
        )
        if not null.num_converged:
            warnings.warn(
                "tgs fit optimizer did not converge with gain rate fixed at "
                "its neutral reference's; gain ratio likelihood-ratio "
                "statistic may be unreliable.",
            )
        lrts[trait] = 2 * (null.fun - free_nll)
    return lrts
