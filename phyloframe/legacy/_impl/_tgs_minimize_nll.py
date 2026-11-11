import typing

import numpy as np
import polars as pl
from scipy import optimize

from ._tgs_calc_nll import tgs_calc_nll


def tgs_pick_selection_rescale(coarsegrain_df: pl.DataFrame) -> float:
    """Implementation detail for `tgs_minimize_nll`.

    Returns the rate by which to rescale the selection coefficient for
    optimization: 1 / mean duration of `coarsegrain_df`'s episodes.
    """
    exposure = pl.col("num_episodes") * pl.col("mean_episode_duration")
    return coarsegrain_df.select(
        pl.col("num_episodes").sum() / exposure.sum()
    ).item()


def tgs_minimize_nll(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    starts: typing.Sequence[typing.Tuple[float, float]],
    fit_gain: bool,
    fit_selection: bool,
    loss_rate: float,
    fit_growth_exponent_bound: float,
) -> optimize.OptimizeResult:
    """Implementation detail for `tgs_fit_trait` and
    `tgs_fit_traits_coarsegrain`.

    Minimizes `tgs_calc_nll(**nll_kwargs, gain_rate=..., growth_rate=...)`
    over log gain rate if `fit_gain` and selection coefficient if
    `fit_selection`, holding the other at its `starts` value.

    Returns the best fit over `starts`, pairs of log gain rate and selection
    coefficient, preferring converged fits, with both parameters in `.x`
    and the number of converged fits in `.num_converged`.

    Notes
    -----
    The selection coefficient is in units of `tgs_pick_selection_rescale`,
    in `starts` and `.x`, so that L-BFGS-B sees parameters of order one. It
    is bounded so that growth exponents stay within
    `fit_growth_exponent_bound`.
    """
    coarsegrain_df = nll_kwargs["coarsegrain_df"]
    selection_rescale = tgs_pick_selection_rescale(coarsegrain_df)
    # longest granule of exposure episodes, by mean calibrated duration
    max_episode_duration = coarsegrain_df["mean_episode_duration"].max()
    max_survival_duration = nll_kwargs["distinct_survival_durations"].max(
        initial=0.0
    )
    # longest time over which a carrier lineage's growth compounds
    max_exponent_time = max_episode_duration + max_survival_duration
    assert np.isfinite(max_exponent_time) and max_exponent_time > 0
    max_selection = (
        fit_growth_exponent_bound / max_exponent_time + loss_rate
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
                (
                    (None, None)
                    if fit_gain
                    else (log_gain_start,) * 2
                ),
                (
                    (None, max_selection)
                    if fit_selection
                    else (selection_start,) * 2
                ),
            ],  # fmt: skip
        )
        for log_gain_start, selection_start in starts
    ]

    converged = [fit for fit in fits if fit.success]
    best = min(converged or fits, key=lambda fit: fit.fun)  # .fun is nll
    best.num_converged = len(converged)
    return best
