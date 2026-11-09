import typing

import numpy as np
import polars as pl
from scipy import optimize

from ._tgs_calc_nll import tgs_calc_nll


def tgs_pick_selection_rescale(coarsened_df: pl.DataFrame) -> float:
    """Implementation detail for `tgs_minimize_nll`.

    Returns the rate by which to rescale the selection coefficient for
    optimization: 1 / mean branch length of `coarsened_df`'s episodes.
    """
    exposure = coarsened_df["num_trials"] * coarsened_df["mean_branch_length"]
    return coarsened_df["num_trials"].sum() / exposure.sum()


def tgs_minimize_nll(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    starts: typing.Sequence[typing.Tuple[float, float]],
    fit_gain: bool,
    fit_selection: bool,
    loss_rate: float,
    growth_exponent_bound: float,
) -> optimize.OptimizeResult:
    """Implementation detail for `tgs_fit_trait` and
    `tgs_fit_coarsened_traits`.

    Minimizes `tgs_calc_nll(**nll_kwargs, gain_rate=..., growth_rate=...)`
    over log gain rate if `fit_gain` and selection coefficient if
    `fit_selection`, holding the other at its `starts` value. Starts from
    each of `starts`, pairs of log gain rate and selection coefficient, and
    returns the best fit, preferring converged ones, with both parameters
    in `.x` and the number of converged fits in `.num_converged`.

    The selection coefficient is optimized, and given in `starts` and `.x`,
    in units of `tgs_pick_selection_rescale`, so that both parameters are
    of order one; otherwise, with long branches, L-BFGS-B stops early on a
    tiny selection coefficient. It is bounded above so that growth
    exponents, growth rate times branch length plus survival duration, stay
    within `growth_exponent_bound`.
    """
    selection_rescale = tgs_pick_selection_rescale(nll_kwargs["coarsened_df"])
    max_exponent_time = nll_kwargs["coarsened_df"][
        "mean_branch_length"
    ].max() + nll_kwargs["distinct_survival_durations"].max(initial=0.0)
    max_selection = (
        growth_exponent_bound / max_exponent_time + loss_rate
    ) / selection_rescale

    def nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        log_gain_rate, selection = params
        res, dlog_gain, drate = tgs_calc_nll(
            **nll_kwargs,
            gain_rate=np.exp(log_gain_rate),
            growth_rate=selection * selection_rescale - loss_rate,
        )
        return res, np.array([dlog_gain, drate * selection_rescale])

    fits = [
        optimize.minimize(
            fun=nll,
            x0=[log_gain_start, selection_start],
            jac=True,  # objective function returns its own gradient
            method="L-BFGS-B",  # gradient-based, with box constraints
            bounds=[
                (None, None) if fit_gain else (log_gain_start,) * 2,
                (
                    (None, max_selection)
                    if fit_selection
                    else (selection_start,) * 2
                ),
            ],
        )
        for log_gain_start, selection_start in starts
    ]
    converged = [fit for fit in fits if fit.success]
    best = min(converged or fits, key=lambda fit: fit.fun)  # .fun is nll
    best.num_converged = len(converged)
    return best
