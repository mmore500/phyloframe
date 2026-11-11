import typing

import numpy as np
import polars as pl
from scipy import sparse, special

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy


def _tgs_calc_effective_length(
    *, length: np.ndarray, growth_rate: float
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for `tgs_calc_nll`.

    Returns effective length (exp(r * t) - 1) / r for each length t, with
    r = `growth_rate`, and its derivative with respect to r.
    """
    cx = growth_rate * length
    # exprel(x) = (exp(x) - 1) / x, finite at x = 0
    effective_length = length * special.exprel(cx)

    # exact derivative, (t * exp(r * t) - effective length) / r, loses
    # precision as r * t approaches 0 (and divides by zero at r = 0), so
    # below a cutoff use the Taylor series t**2 / 2 + r * t**3 / 3 + O(r**2)
    # see https://en.wikipedia.org/wiki/Exponential_function#Power_series
    is_near_zero_rate = np.abs(cx) < 1e-6
    series_deriv = length * length * (0.5 + cx / 3)
    exact_deriv = np.divide(
        length * np.exp(cx) - effective_length,
        growth_rate,
        out=np.empty_like(cx),
        where=~is_near_zero_rate,
    )
    return effective_length, np.where(
        is_near_zero_rate, series_deriv, exact_deriv
    )


def tgs_calc_nll(
    *,
    gain_rate: float,
    growth_rate: float,
    coarsegrain_df: pl.DataFrame,
    survival_duration_granule_mass: sparse.csr_matrix,
    distinct_survival_durations: np.ndarray,
) -> typing.Tuple[float, float, float]:
    """Implementation detail for `tgs_minimize_nll`.

    Returns negative log likelihood (up to a constant) of one trait's
    coarsegrained exposure episodes, and its derivatives with respect to log
    gain rate and to `growth_rate`, r, the carrier growth rate (selection
    coefficient minus loss rate). An episode of length `t` gains the trait
    with probability

        gain_rate * (exp(r * t) - 1) / r * exp(r * u)

    where `u` is its survival duration, the time until its clade's last leaf
    is sampled (0 for a leaf's pendant branch). Each row of `coarsegrain_df`
    averages exp(r * u) over its episodes, weighted by its row of
    `survival_duration_granule_mass` over `distinct_survival_durations`; see
    `_tgs_coarsegrain_episodes`.

    Input Schema
    ------------
    `coarsegrain_df` columns:

    'num_episodes' : float
        (Weighted) count of the row's exposure episodes.
    'num_gains' : float
        Of those, the (weighted) count that gained the trait.
    'mean_episode_duration' : float
        Their average calibrated length.
    """
    effective_length, effective_length_deriv = _tgs_calc_effective_length(
        length=coarsegrain_df["mean_episode_duration"].to_numpy(),
        growth_rate=growth_rate,
    )

    # carrier lineage growth over each distinct survival duration, exp(r * u),
    # then averaged over each coarsegrained row's episodes
    growth_by_duration = np.exp(growth_rate * distinct_survival_durations)
    row_survival = survival_duration_granule_mass @ growth_by_duration
    row_survival_deriv = survival_duration_granule_mass @ (
        distinct_survival_durations * growth_by_duration
    )

    gain_prob = gain_rate * effective_length * row_survival
    nll, dnll_dgain_prob = calc_binomial_nll_proxy(
        coarsegrain_df["num_episodes"].to_numpy(),
        coarsegrain_df["num_gains"].to_numpy(),
        gain_prob,
    )

    # chain rule, as d gain_prob / d log(gain_rate) = gain_prob
    dnll_dlog_gain = np.sum(dnll_dgain_prob * gain_prob)
    dnll_drate = np.sum(
        dnll_dgain_prob
        * gain_rate
        * (
            effective_length_deriv * row_survival
            + effective_length * row_survival_deriv
        )
    )

    return nll, dnll_dlog_gain, dnll_drate
