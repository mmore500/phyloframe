import math
import typing
import warnings

import numpy as np
import polars as pl

from ..._auxlib._clamp import clamp
from ._tgs_fit_trait import tgs_fit_trait
from ._tgs_minimize_nll import tgs_minimize_nll, tgs_pick_selection_rescale


def _tgs_calc_lrt(*, free_nll: float, **null_kwargs) -> float:
    """Implementation detail for `tgs_fit_coarsened_traits`.

    Returns the likelihood-ratio statistic, twice the increase in negative
    log likelihood from `free_nll` to that of the null fit,
    `tgs_minimize_nll(**null_kwargs)`, clipped at 0.
    """
    null = tgs_minimize_nll(**null_kwargs)
    is_null_better = null.fun < free_nll and not math.isclose(
        null.fun, free_nll, rel_tol=1e-8, abs_tol=1e-8
    )
    if not null.num_converged or is_null_better:
        warnings.warn(
            "tgs fit optimizer did not converge under the null hypothesis, "
            "or found a worse free fit than null fit; likelihood-ratio "
            "statistic may be unreliable.",
        )
    return clamp(2 * (null.fun - free_nll), lo=0.0)


def tgs_fit_coarsened_traits(
    *,
    trait_nll_kwargs: typing.Sequence[typing.Dict[str, typing.Any]],
    neutral_trait_ids: typing.Sequence[int],
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    progress_wrap: typing.Callable,
) -> pl.DataFrame:
    """Implementation detail for `tgs_fit_traits`.

    Fits each trait's gain rate and selection coefficient, given each
    trait's `tgs_calc_nll` arguments but the rate parameters. Returns one
    row per trait.

    Output Schema
    -------------
    'gain_rate' : float
        Fitted gain rate, per unit episode length; 0 without gains.
    'selection_coef' : float
        Fitted selection coefficient; nan without gains.
    'gain_ratio' : float
        Gain rate relative to the mean gain rate of the other neutral
        traits; nan if they have no gains.
    'selection_coef_lrt_naive' : float
        Likelihood-ratio statistic for selection coefficient = 0, refitting
        the gain rate; nan without gains.
    'gain_ratio_lrt_naive' : float
        Likelihood-ratio statistic for gain ratio = 1, refitting the
        selection coefficient; nan without gains or gain ratio.
    """
    minimize_kwargs = dict(
        loss_rate=loss_rate, growth_exponent_bound=growth_exponent_bound
    )
    fits = [
        tgs_fit_trait(
            nll_kwargs=nll_kwargs,
            selection_coef_starts=selection_coef_starts,
            **minimize_kwargs,
        )
        for nll_kwargs in progress_wrap(trait_nll_kwargs)
    ]
    gain_rates = np.array([fit["gain_rate"] for fit in fits])

    def calc_trait_estimates(trait_id: int) -> typing.Dict[str, float]:
        nll_kwargs, fit = trait_nll_kwargs[trait_id], fits[trait_id]
        # a neutral trait is not compared against itself
        reference_trait_ids = sorted(set(neutral_trait_ids) - {trait_id})
        reference_gain_rate = gain_rates[reference_trait_ids].mean()
        if not reference_gain_rate > 0:  # no reference trait has gains
            reference_gain_rate = np.nan
        res = dict(
            gain_rate=fit["gain_rate"],
            selection_coef=fit["selection_coef"],
            gain_ratio=fit["gain_rate"] / reference_gain_rate,
            selection_coef_lrt_naive=np.nan,
            gain_ratio_lrt_naive=np.nan,
        )
        if not fit["gain_rate"] > 0:
            return res  # selection is undefined without gains
        lrt_kwargs = dict(
            free_nll=fit["nll"], nll_kwargs=nll_kwargs, **minimize_kwargs
        )
        # null of no selection: one start suffices, as the negative log
        # likelihood is then convex in log gain rate
        res["selection_coef_lrt_naive"] = _tgs_calc_lrt(
            starts=[(np.log(fit["gain_rate"]), 0.0)],
            fit_gain=True,
            fit_selection=False,
            **lrt_kwargs,
        )
        if np.isfinite(reference_gain_rate):
            # null of gain rate equal to the neutral reference's, taken as
            # known; start from the free fit, too
            selection_rescale = tgs_pick_selection_rescale(
                nll_kwargs["coarsened_df"]
            )
            free_start = fit["selection_coef"] / selection_rescale
            res["gain_ratio_lrt_naive"] = _tgs_calc_lrt(
                starts=[
                    (np.log(reference_gain_rate), start)
                    for start in (free_start, *selection_coef_starts)
                ],
                fit_gain=False,
                fit_selection=True,
                **lrt_kwargs,
            )
        return res

    return pl.DataFrame(
        [calc_trait_estimates(trait_id) for trait_id in range(len(fits))],
    )
