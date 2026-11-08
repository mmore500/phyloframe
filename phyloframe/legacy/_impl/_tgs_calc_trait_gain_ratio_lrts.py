import typing
import warnings

import numpy as np
import polars as pl

from ._tgs_calc_nll import tgs_calc_nll
from ._tgs_minimize_nll import tgs_minimize_nll, tgs_pick_selection_rescale


def _tgs_calc_trait_gain_ratio_lrt(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    gain_rate_trait: float,
    gain_rate_neutral: float,
    selection_coef: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
) -> float:
    """Implementation detail for `tgs_calc_trait_gain_ratio_lrts`. Acronym
    tgs stands for trait gain/selection.

    Returns one trait's gain ratio likelihood-ratio statistic (see
    `tgs_calc_trait_gain_ratio_lrts`), given its fitted gain rate,
    `gain_rate_trait`, and selection coefficient, and the mean gain rate of
    its neutral reference, `gain_rate_neutral`; nan if its gain ratio is
    undefined.
    """
    if not (gain_rate_trait > 0 and np.isfinite(gain_rate_neutral)):
        return np.nan
    selection_rescale = tgs_pick_selection_rescale(nll_kwargs["coarsened_df"])
    free_nll, __, __ = tgs_calc_nll(
        **nll_kwargs,
        gain_rate=gain_rate_trait,
        growth_rate=selection_coef - loss_rate,
    )
    null = tgs_minimize_nll(
        nll_kwargs=nll_kwargs,
        # from the free fit, then from each of `selection_coef_starts`
        starts=[
            (np.log(gain_rate_neutral), start)
            for start in (
                selection_coef / selection_rescale,
                *selection_coef_starts,
            )
        ],
        fit_gain=False,
        fit_selection=True,
        selection_rescale=selection_rescale,
        loss_rate=loss_rate,
    )
    if not null.num_converged:
        warnings.warn(
            "tgs fit optimizer did not converge with gain rate fixed at the "
            "mean gain rate of its neutral reference traits; gain ratio "
            "likelihood-ratio statistic may be unreliable.",
        )
    return float(2 * (null.fun - free_nll))


def tgs_calc_trait_gain_ratio_lrts(
    *,
    fits_df: pl.DataFrame,
    trait_nll_kwargs: typing.Sequence[typing.Dict[str, typing.Any]],
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns, per trait, the likelihood-ratio test (LRT) statistic for gain
    ratio = 1, a profile likelihood test: twice the increase in negative
    log likelihood from holding the trait's gain rate at the mean fitted
    gain rate of its neutral reference, `gain_rate_neutral`,
    and refitting the selection coefficient alone; see Output Schema.
    `fits_df` holds each trait's fitted `gain_rate` and `selection_coef`,
    and its neutral reference's `gain_rate_neutral`, and `trait_nll_kwargs`
    each trait's `tgs_calc_nll`
    arguments but the rate parameters.

    The statistic naively treats the reference gain rate as known, and
    exposure episodes as independent, so overstates significance.

    Input Schema
    ------------
    `fits_df` columns, one row per trait, e.g., from `tgs_fit_trait`:

    'gain_rate' : float
        Fitted gain rate.
    'selection_coef' : float
        Fitted selection coefficient.
    'gain_rate_neutral' : float
        Mean gain rate of the trait's neutral reference; nan if undefined
        (see `tgs_calc_neutral_reference_gain_rates`).

    Output Schema
    -------------
    One row per trait, in `fits_df` row order:

    'gain_ratio_lrt_naive' : float
        The statistic; nan if the gain ratio is undefined, or the trait has
        no gains.
    """
    return pl.DataFrame(
        {
            "gain_ratio_lrt_naive": [
                _tgs_calc_trait_gain_ratio_lrt(
                    nll_kwargs=nll_kwargs,
                    gain_rate_trait=fit["gain_rate"],
                    gain_rate_neutral=fit["gain_rate_neutral"],
                    selection_coef=fit["selection_coef"],
                    loss_rate=loss_rate,
                    selection_coef_starts=selection_coef_starts,
                )
                for nll_kwargs, fit in zip(
                    trait_nll_kwargs, fits_df.iter_rows(named=True)
                )
            ],
        },
        schema={"gain_ratio_lrt_naive": pl.Float64},
    )
