import typing
import warnings

import numpy as np
import pandas as pd

from ._tgs_fit import _tgs_calc_scales, tgs_minimize_nll


def _tgs_calc_gain_ratio_lrt(
    *,
    partitions: pd.DataFrame,
    calc_nll: typing.Callable,
    gain_rate: float,
    selection_coef: float,
    gain_ratio: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
) -> float:
    """Implementation detail for `tgs_calc_gain_ratio_lrts`. Acronym tgs
    stands for trait gain/selection.

    Returns one trait's gain ratio likelihood-ratio statistic (see
    `tgs_calc_gain_ratio_lrts`), or nan if its gain ratio is undefined.
    """
    if not np.isfinite(gain_ratio) or not gain_rate > 0:
        return np.nan
    selection_scale, __ = _tgs_calc_scales(partitions)
    free_nll, __, __ = calc_nll(
        log_gain_rate=np.log(gain_rate), rate=selection_coef - loss_rate
    )
    reference_log_gain_rate = np.log(gain_rate / gain_ratio)
    null = tgs_minimize_nll(
        calc_nll=calc_nll,
        # from the free fit, then from each of `selection_coef_starts`
        starts=[
            (reference_log_gain_rate, start)
            for start in (
                selection_coef / selection_scale,
                *selection_coef_starts,
            )
        ],
        fit_gain=False,
        fit_selection=True,
        selection_scale=selection_scale,
        loss_rate=loss_rate,
    )
    if not null.num_converged:
        warnings.warn(
            "tgs fit optimizer did not converge with gain rate fixed at the "
            "mean gain rate of its neutral reference traits; gain ratio "
            "likelihood-ratio statistic may be unreliable.",
        )
    return 2 * (null.fun - free_nll)


def tgs_calc_gain_ratio_lrts(
    *,
    fits_df: pd.DataFrame,
    trait_partitions: typing.Sequence[pd.DataFrame],
    calc_nlls: typing.Sequence[typing.Callable],
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns, per trait, the likelihood-ratio test (LRT) statistic for gain
    ratio = 1, a profile likelihood test: twice the increase in negative
    log likelihood from holding the trait's gain rate at the mean fitted
    gain rate of its neutral reference, i.e., `gain_rate / gain_ratio`, and
    refitting the selection coefficient alone. `fits_df` holds each trait's
    fitted `gain_rate`, `selection_coef`, and `gain_ratio`;
    `trait_partitions`, each trait's partitions (see `tgs_fit_partitions`);
    and `calc_nlls`, their `calc_nll(log_gain_rate=..., rate=...)` (see
    `tgs_calc_nll`).

    The statistic naively treats the reference gain rate as known, and
    exposure episodes (or merged rows of them) as independent, so overstates significance. It is
    nan wherever the gain ratio is undefined, or the trait has no gains.
    """
    return np.array(
        [
            _tgs_calc_gain_ratio_lrt(
                partitions=partitions,
                calc_nll=calc_nll,
                gain_rate=fit.gain_rate,
                selection_coef=fit.selection_coef,
                gain_ratio=fit.gain_ratio,
                loss_rate=loss_rate,
                selection_coef_starts=selection_coef_starts,
            )
            for partitions, calc_nll, fit in zip(
                trait_partitions, calc_nlls, fits_df.itertuples()
            )
        ],
        dtype=np.float64,
    )
