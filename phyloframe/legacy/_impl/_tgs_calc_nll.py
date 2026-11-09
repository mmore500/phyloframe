import typing

import numpy as np
import polars as pl
from scipy import sparse, special

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy


def _tgs_calc_effective_length(
    *, length: np.ndarray, growth_rate: float
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for `tgs_calc_nll`. Acronym tgs stands for trait
    gain/selection.

    Returns effective length (exp(r * t) - 1) / r for each length t, with
    r = `growth_rate`, and its derivative with respect to r.
    """
    cx = growth_rate * length
    # exprel(x) = (exp(x) - 1) / x, finite at x = 0
    effective_length = length * special.exprel(cx)

    # The exact derivative, (t * exp(r * t) - effective length) / r,
    # loses precision as r * t approaches 0, as it divides a difference of
    # nearly equal numbers by a small r (relative error ~1e-16 /
    # |r * t|). In this regime, the Taylor series of the exponential
    # function provides better numerical stability: with effective length =
    # sum over j >= 1 of r**(j - 1) * t**j / j!, the derivative is
    # t**2 / 2 + r * t**3 / 3 + O(r**2), truncated with relative error
    # ~(r * t)**2 / 4; see
    # https://en.wikipedia.org/wiki/Exponential_function#Power_series
    #
    # So, use the series below a |r * t| < 1e-6 cutoff, where it is more
    # accurate (~1e-13 vs ~1e-10 error at the cutoff, and the exact form
    # divides by zero at r = 0).
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
    coarsened_df: pl.DataFrame,
    survival_shares: sparse.csr_matrix,
    distinct_survival_durations: np.ndarray,
) -> typing.Tuple[float, float, float]:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns negative log likelihood (up to a constant) of one trait's
    `coarsened_df` (see Input Schema), and its derivatives with respect to log
    gain rate and to `growth_rate`, r, the carrier growth rate (selection
    coefficient minus loss rate). A branch of length `t` gains the trait
    with probability

        gain_rate * (exp(r * t) - 1) / r * exp(r * u)

    where `u` is its survival duration, the time until its clade's last leaf
    is sampled, e.g., 0 for a leaf's pendant branch. Each coarsened row
    averages exp(r * u) exactly over its branches, whose shares over
    `distinct_survival_durations` are rows of `survival_shares`. Requires
    r * (t + u) to stay well below float64 overflow, as `tgs_minimize_nll`
    ensures.

    Input Schema
    ------------
    `coarsened_df` columns, e.g., from `_tgs_coarsen_episodes`:

    'num_trials' : float
        (Weighted) count of the coarsened row's exposure episodes.
    'num_successes' : float
        Of those, the (weighted) count that gained the trait.
    'length' : float
        Their mean branch length.
    """
    effective_length, effective_length_deriv = _tgs_calc_effective_length(
        length=coarsened_df["length"].to_numpy(),
        growth_rate=growth_rate,
    )

    # chance a carrier lineage persists until its clade's last leaf is
    # sampled, averaged over each coarsened row's branches
    survival_factors = np.exp(growth_rate * distinct_survival_durations)
    survival = survival_shares @ survival_factors
    survival_deriv = survival_shares @ (
        distinct_survival_durations * survival_factors
    )

    p = gain_rate * effective_length * survival
    nll, dnll_dp = calc_binomial_nll_proxy(
        coarsened_df["num_trials"].to_numpy(),
        coarsened_df["num_successes"].to_numpy(),
        p,
    )

    # chain rule, as d p / d log(gain_rate) = p
    dnll_dlog_gain = np.sum(dnll_dp * p)
    dnll_drate = np.sum(
        dnll_dp
        * gain_rate
        * (
            effective_length_deriv * survival
            + effective_length * survival_deriv
        )
    )

    return nll, dnll_dlog_gain, dnll_drate
