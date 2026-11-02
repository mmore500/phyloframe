import typing

import numpy as np
import pandas as pd
from scipy import sparse, special

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy
from ._tgs_fit import _tgs_calc_effective_length_deriv


def tgs_calc_allmdc_nll(
    *,
    log_gain_rate: float,
    rate: float,
    partitions: pd.DataFrame,
    survival_shares: sparse.csr_matrix,
    distinct_survival_times: np.ndarray,
    growth_exponent_bound: float,
) -> typing.Tuple[float, float, float]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns negative log likelihood (up to a constant) of one trait's
    `partitions` and their `survival_shares` (see `tgs_calc_allmdc_data`),
    and its derivatives with respect to log gain rate and to `rate`
    (selection minus loss rate). A branch of length `t` with survival time
    `u` gains the trait, i.e., its whole clade carries it, with probability

        gain_rate * (exp(rate * t) - 1) / rate * exp(rate * u)

    averaging exp(rate * u) exactly over each partition's branches, whose
    shares over `distinct_survival_times` are rows of `survival_shares`,
    with exponents capped at `growth_exponent_bound`.
    """
    num_trials = partitions["num_trials"].to_numpy()
    num_successes = partitions["num_successes"].to_numpy()
    lengths = partitions["length"].to_numpy()
    gain_rate = np.exp(log_gain_rate)
    effective_length = lengths * special.exprel(
        np.minimum(rate * lengths, growth_exponent_bound)
    )
    effective_length_deriv = _tgs_calc_effective_length_deriv(
        pendant_length=lengths,
        rate=rate,
        growth_exponent_bound=growth_exponent_bound,
    )
    # chance a carrier lineage persists until its clade's last leaf is
    # sampled, averaged over each partition's branches
    survival_factors = np.exp(
        np.minimum(rate * distinct_survival_times, growth_exponent_bound)
    )
    survival = survival_shares @ survival_factors
    survival_deriv = survival_shares @ (
        distinct_survival_times * survival_factors
    )
    p = gain_rate * effective_length * survival
    nll, dnll_dp = calc_binomial_nll_proxy(num_trials, num_successes, p)
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
