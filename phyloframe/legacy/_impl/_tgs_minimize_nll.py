import typing

import numpy as np
import polars as pl
from scipy import optimize

from ._tgs_calc_nll import tgs_calc_nll


def _tgs_calc_exposure(coarsened_df: pl.DataFrame) -> float:
    """Implementation detail for `tgs_minimize_nll` callers. Acronym tgs
    stands for trait gain/selection.

    Returns the total branch length of `coarsened_df`' episodes.
    """
    return (coarsened_df["num_trials"] * coarsened_df["length"]).sum()


def tgs_pick_selection_rescale(coarsened_df: pl.DataFrame) -> float:
    """Implementation detail for `tgs_minimize_nll` callers. Acronym tgs
    stands for trait gain/selection.

    Picks the rate by which to rescale the selection coefficient for
    optimization (see `tgs_minimize_nll`): 1 / mean branch length of
    `coarsened_df`' episodes, a typical rate.
    """
    return coarsened_df["num_trials"].sum() / _tgs_calc_exposure(coarsened_df)


def tgs_pick_log_gain_rate_start(coarsened_df: pl.DataFrame) -> float:
    """Implementation detail for `tgs_minimize_nll` callers. Acronym tgs
    stands for trait gain/selection.

    Picks the starting point for fitting the log gain rate:
    log(gains per unit branch length) of `coarsened_df`, the log gain rate
    under no selection.
    """
    return np.log(
        coarsened_df["num_successes"].sum() / _tgs_calc_exposure(coarsened_df)
    )


def tgs_minimize_nll(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    starts: typing.Sequence[typing.Tuple[float, float]],
    fit_gain: bool,
    fit_selection: bool,
    selection_rescale: float,
    loss_rate: float,
) -> optimize.OptimizeResult:
    """Implementation detail for `tgs_fit_trait`,
    `_tgs_calc_trait_selection_lrts`, and `_tgs_calc_trait_gain_ratio_lrts`.
    Acronym tgs stands for trait gain/selection.

    Minimizes `tgs_calc_nll(**nll_kwargs, gain_rate=..., growth_rate=...)` by
    L-BFGS-B over log gain rate if `fit_gain` and selection coefficient if
    `fit_selection`, holding the others at their `starts` values. Starts from
    each of `starts` and returns the best fit, preferring converged ones, with
    both parameters in `.x` and the number of converged fits in
    `.num_converged`.

    The selection coefficient is optimized, and given in `starts`, as a
    multiple of `selection_rescale`, a typical rate, e.g., 1 / mean branch
    length, so that both parameters are of order one; otherwise, with long
    lengths, L-BFGS-B stops early on the tiny selection coefficient. It is
    bounded above so that growth exponents, growth rate times branch length
    plus survival duration, stay within `growth_exponent_bound` of
    `nll_kwargs` (see `tgs_calc_nll`), and starts beyond are clipped.
    """
    is_free = np.array([fit_gain, fit_selection])
    durations = nll_kwargs["distinct_survival_durations"]
    max_exponent_time = nll_kwargs["coarsened_df"]["length"].max() + (
        0.0 if durations is None else durations.max(initial=0.0)
    )
    max_rescaled_selection = (
        (nll_kwargs["growth_exponent_bound"] / max_exponent_time + loss_rate)
        / selection_rescale
        if max_exponent_time > 0
        else None
    )
    bounds = [(None, None), (None, max_rescaled_selection)]

    def nll(
        free_params: np.ndarray, start: np.ndarray
    ) -> typing.Tuple[float, np.ndarray]:
        params = start.copy()
        params[is_free] = free_params
        log_gain_rate, rescaled_selection = params
        res, dlog_gain, drate = tgs_calc_nll(
            **nll_kwargs,
            gain_rate=np.exp(log_gain_rate),
            growth_rate=rescaled_selection * selection_rescale - loss_rate,
        )
        return res, np.array([dlog_gain, drate * selection_rescale])[is_free]

    fits = []
    for start in starts:
        start = np.asarray(start, dtype=np.float64)
        fit = optimize.minimize(
            nll,
            start[is_free],
            args=(start,),
            jac=True,  # objective function returns its own gradient
            method="L-BFGS-B",  # gradient-based, for differentiable functions
            bounds=[bound for bound, free in zip(bounds, is_free) if free],
        )
        params = start.copy()
        params[is_free] = fit.x
        fit.x = params
        fits.append(fit)
    converged = [fit for fit in fits if fit.success]
    best = min(converged or fits, key=lambda fit: fit.fun)  # .fun is nll
    best.num_converged = len(converged)
    return best
