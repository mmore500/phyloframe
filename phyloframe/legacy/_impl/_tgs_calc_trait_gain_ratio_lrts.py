import typing
import warnings

import numpy as np
import polars as pl

from ._tgs_calc_nll import tgs_calc_nll
from ._tgs_minimize_nll import _tgs_calc_scales, tgs_minimize_nll


def _tgs_calc_trait_gain_ratio_lrt(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    gain_rate: float,
    selection_coef: float,
    gain_ratio: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
) -> float:
    """Implementation detail for `tgs_calc_trait_gain_ratio_lrts`. Acronym
    tgs stands for trait gain/selection.

    Returns one trait's gain ratio likelihood-ratio statistic (see
    `tgs_calc_trait_gain_ratio_lrts`), or nan if its gain ratio is
    undefined.
    """
    if not np.isfinite(gain_ratio) or not gain_rate > 0:
        return np.nan
    selection_scale, __ = _tgs_calc_scales(nll_kwargs["partitions"])
    free_nll, __, __ = tgs_calc_nll(
        **nll_kwargs,
        log_gain_rate=np.log(gain_rate),
        rate=selection_coef - loss_rate,
    )
    reference_log_gain_rate = np.log(gain_rate / gain_ratio)
    null = tgs_minimize_nll(
        nll_kwargs=nll_kwargs,
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

    Returns, per `trait` (index), the likelihood-ratio test (LRT) statistic
    for gain ratio = 1, `gain_ratio_lrt_naive`, a profile likelihood test:
    twice the increase in negative log likelihood from holding the trait's
    gain rate at the mean fitted gain rate of its neutral reference, i.e.,
    `gain_rate / gain_ratio`, and refitting the selection coefficient
    alone. `fits_df` holds each trait's fitted `gain_rate`,
    `selection_coef`, and `gain_ratio`, and `trait_nll_kwargs` each trait's
    `tgs_calc_nll` arguments but the rate parameters.

    The statistic naively treats the reference gain rate as known, and
    exposure episodes (or merged rows of them) as independent, so
    overstates significance. It is nan wherever the gain ratio is
    undefined, or the trait has no gains.
    """
    return pl.DataFrame(
        [
            {
                "trait": trait,
                "gain_ratio_lrt_naive": _tgs_calc_trait_gain_ratio_lrt(
                    nll_kwargs=nll_kwargs,
                    gain_rate=fit["gain_rate"],
                    selection_coef=fit["selection_coef"],
                    gain_ratio=fit["gain_ratio"],
                    loss_rate=loss_rate,
                    selection_coef_starts=selection_coef_starts,
                ),
            }
            for trait, (nll_kwargs, fit) in enumerate(
                zip(trait_nll_kwargs, fits_df.iter_rows(named=True))
            )
        ],
        schema={"trait": pl.Int64, "gain_ratio_lrt_naive": pl.Float64},
    )
