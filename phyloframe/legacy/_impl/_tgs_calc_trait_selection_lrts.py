import typing
import warnings

import numpy as np
import polars as pl

from ._tgs_calc_nll import tgs_calc_nll
from ._tgs_minimize_nll import tgs_calc_selection_scale, tgs_minimize_nll


def tgs_calc_trait_selection_lrt(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    gain_rate: float,
    selection_coef: float,
    loss_rate: float,
) -> float:
    """Implementation detail for `tgs_fit` and
    `tgs_calc_trait_selection_lrts`. Acronym tgs stands for trait
    gain/selection.

    Returns one trait's selection likelihood-ratio statistic (see
    `tgs_calc_trait_selection_lrts`), given its fitted `gain_rate` and
    `selection_coef`, or nan if it has no gains.
    """
    if not gain_rate > 0:
        return np.nan
    selection_scale = tgs_calc_selection_scale(nll_kwargs["partitions"])
    free_nll, __, __ = tgs_calc_nll(
        **nll_kwargs,
        log_gain_rate=np.log(gain_rate),
        rate=selection_coef - loss_rate,
    )
    # under the null hypothesis of no selection, refit the log gain rate
    # alone, with selection held at 0; one start, from the free fit,
    # suffices, as the negative log likelihood is then convex in log gain
    # rate (each bin's gain probability is gain rate times a fixed exposure)
    null = tgs_minimize_nll(
        nll_kwargs=nll_kwargs,
        starts=[(np.log(gain_rate), 0.0)],  # (log gain rate, selection)
        fit_gain=True,
        fit_selection=False,
        selection_scale=selection_scale,
        loss_rate=loss_rate,
    )
    if not null.num_converged:
        warnings.warn(
            "tgs fit optimizer did not converge under the null hypothesis "
            f"of no selection ({null.message}); likelihood-ratio statistic "
            "may be unreliable.",
        )
    # two-tailed, testing selection coefficient != 0 in either direction
    return float(2 * (null.fun - free_nll))


def tgs_calc_trait_selection_lrts(
    *,
    fits_df: pl.DataFrame,
    trait_nll_kwargs: typing.Sequence[typing.Dict[str, typing.Any]],
    loss_rate: float,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns, per trait, the likelihood-ratio test (LRT) statistic for
    selection coefficient = 0: twice the increase in negative log
    likelihood from fixing the selection coefficient at 0 and refitting
    the gain rate alone; see Output Schema. `fits_df` holds each trait's
    fitted `gain_rate` and `selection_coef`, and `trait_nll_kwargs` each
    trait's `tgs_calc_nll` arguments but the rate parameters.

    The statistic naively treats exposure episodes (or merged rows of
    them) as independent, so overstates significance.

    Output Schema
    -------------
    'trait' : integer
        Trait index.
    'selection_coef_lrt_naive' : float
        The statistic; nan if the trait has no gains.
    """
    return pl.DataFrame(
        [
            {
                "trait": trait,
                "selection_coef_lrt_naive": tgs_calc_trait_selection_lrt(
                    nll_kwargs=nll_kwargs,
                    gain_rate=fit["gain_rate"],
                    selection_coef=fit["selection_coef"],
                    loss_rate=loss_rate,
                ),
            }
            for trait, (nll_kwargs, fit) in enumerate(
                zip(trait_nll_kwargs, fits_df.iter_rows(named=True))
            )
        ],
        schema={"trait": pl.Int64, "selection_coef_lrt_naive": pl.Float64},
    )
