import typing

import numpy as np
import polars as pl
from scipy import optimize

from ._tgs_calc_nll import tgs_calc_nll


def _tgs_calc_scales(partitions: pl.DataFrame) -> typing.Tuple[float, float]:
    """Implementation detail for `tgs_minimize_nll` callers. Acronym tgs
    stands for trait gain/selection.

    Returns the selection scale, 1 / mean branch length, and the neutral
    log gain rate, log(gains per unit branch length), of `partitions`.
    """
    num_trials = partitions["num_trials"].to_numpy()
    exposure = np.sum(num_trials * partitions["length"].to_numpy())
    return (
        num_trials.sum() / exposure,
        np.log(partitions["num_successes"].sum() / exposure),
    )


def tgs_minimize_nll(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    starts: typing.Sequence[typing.Tuple[float, float]],
    fit_gain: bool,
    fit_selection: bool,
    selection_scale: float,
    loss_rate: float,
) -> optimize.OptimizeResult:
    """Implementation detail for `tgs_fit_partitions`,
    `tgs_calc_trait_selection_lrts`, and `tgs_calc_trait_gain_ratio_lrts`.
    Acronym tgs stands for trait gain/selection.

    Minimizes `tgs_calc_nll(**nll_kwargs, log_gain_rate=..., rate=...)` by
    L-BFGS-B over log gain rate if `fit_gain` and selection coefficient if
    `fit_selection`, holding the others at their `starts` values. Starts from
    each of `starts` and returns the best fit, preferring converged ones, with
    both parameters in `.x` and the number of converged fits in
    `.num_converged`.

    The selection coefficient is optimized, and given in `starts`, as a
    multiple of `selection_scale`, a typical rate, e.g., 1 / mean branch
    length, so that both parameters are of order one; otherwise, with long
    lengths, L-BFGS-B stops early on the tiny selection coefficient.
    """
    is_free = np.array([fit_gain, fit_selection])

    def nll(
        free_params: np.ndarray, start: np.ndarray
    ) -> typing.Tuple[float, np.ndarray]:
        params = start.copy()
        params[is_free] = free_params
        log_gain_rate, scaled_selection = params
        res, dlog_gain, drate = tgs_calc_nll(
            **nll_kwargs,
            log_gain_rate=log_gain_rate,
            rate=scaled_selection * selection_scale - loss_rate,
        )
        return res, np.array([dlog_gain, drate * selection_scale])[is_free]

    fits = []
    for start in starts:
        start = np.asarray(start, dtype=np.float64)
        fit = optimize.minimize(
            nll,
            start[is_free],
            args=(start,),
            jac=True,  # objective function returns its own gradient
            method="L-BFGS-B",  # gradient-based, for differentiable functions
        )
        params = start.copy()
        params[is_free] = fit.x
        fit.x = params
        fits.append(fit)
    converged = [fit for fit in fits if fit.success]
    best = min(converged or fits, key=lambda fit: fit.fun)  # .fun is nll
    best.num_converged = len(converged)
    return best
