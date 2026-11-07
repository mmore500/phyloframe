import typing

import numpy as np
import polars as pl
from scipy import sparse, special

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy


def _tgs_calc_effective_length(
    *, length: np.ndarray, rate: float, growth_exponent_bound: float
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Implementation detail for `tgs_calc_nll`. Acronym tgs stands for trait
    gain/selection.

    Returns effective length (exp(rate * t) - 1) / rate for each length t,
    and its derivative with respect to `rate`, with rate * t capped at
    `growth_exponent_bound`.
    """
    cx = np.minimum(rate * length, growth_exponent_bound)
    # exprel(x) = (exp(x) - 1) / x, finite at x = 0
    effective_length = length * special.exprel(cx)

    # The exact derivative, (t * exp(rate * t) - effective length) / rate,
    # loses precision as rate * t approaches 0, as it divides a difference of
    # nearly equal numbers by a small rate (relative error ~1e-16 /
    # |rate * t|). In this regime, the Taylor series of the exponential
    # function provides better numerical stability: with effective length =
    # sum over j >= 1 of rate**(j - 1) * t**j / j!, the derivative is
    # t**2 / 2 + rate * t**3 / 3 + O(rate**2), truncated with relative error
    # ~(rate * t)**2 / 4; see
    # https://en.wikipedia.org/wiki/Exponential_function#Power_series
    #
    # So, use the series below a |rate * t| < 1e-6 cutoff, where it is more
    # accurate (~1e-13 vs ~1e-10 error at the cutoff, and the exact form
    # divides by zero at rate = 0).
    is_near_zero_rate = np.abs(cx) < 1e-6
    series_deriv = length * length * (0.5 + cx / 3)
    exact_deriv = np.divide(
        length * np.exp(cx) - effective_length,
        rate,
        out=np.empty_like(cx),
        where=~is_near_zero_rate,
    )
    return effective_length, np.where(
        is_near_zero_rate, series_deriv, exact_deriv
    )


def tgs_calc_nll(
    *,
    log_gain_rate: float,
    rate: float,
    partitions: pl.DataFrame,
    growth_exponent_bound: float,
    survival_shares: typing.Optional[sparse.csr_matrix] = None,
    distinct_survival_durations: typing.Optional[np.ndarray] = None,
) -> typing.Tuple[float, float, float]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns negative log likelihood (up to a constant) of one trait's
    `partitions` (see Input Schema), and its derivatives with respect to log
    gain rate and to `rate`
    (selection coefficient minus loss rate). A branch of length `t` gains the
    trait with probability

        gain_rate * (exp(rate * t) - 1) / rate * exp(rate * u)

    where `u` is its survival duration, the time until its clade's last leaf
    is sampled. Each partition averages exp(rate * u) exactly over its
    branches, whose shares over `distinct_survival_durations` are rows of
    `survival_shares`. Without `survival_shares`, `u` is 0, as for leaf
    pendant branches. Exponents are capped at `growth_exponent_bound`.

    Input Schema
    ------------
    `partitions` columns, e.g., from `tgs_merge_episodes`:

    'num_trials' : float
        (Weighted) count of the partition's exposure episodes.
    'num_successes' : float
        Of those, the (weighted) count that gained the trait.
    'length' : float
        Their mean branch length.
    """
    if not growth_exponent_bound < np.log(np.finfo(np.float64).max):
        raise ValueError("growth exponent bound overflows float64")
    num_trials = partitions["num_trials"].to_numpy()
    num_successes = partitions["num_successes"].to_numpy()
    lengths = partitions["length"].to_numpy()
    gain_rate = np.exp(log_gain_rate)
    effective_length, effective_length_deriv = _tgs_calc_effective_length(
        length=lengths, rate=rate, growth_exponent_bound=growth_exponent_bound
    )
    if survival_shares is None:
        survival, survival_deriv = 1.0, 0.0
    else:
        # chance a carrier lineage persists until its clade's last leaf is
        # sampled, averaged over each partition's branches
        survival_factors = np.exp(
            np.minimum(
                rate * distinct_survival_durations, growth_exponent_bound
            )
        )
        survival = survival_shares @ survival_factors
        survival_deriv = survival_shares @ (
            distinct_survival_durations * survival_factors
        )
    p = gain_rate * effective_length * survival
    nll, dnll_dp = calc_binomial_nll_proxy(num_trials, num_successes, p)
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
