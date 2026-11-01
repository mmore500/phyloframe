import typing

import numpy as np
from scipy import special

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy
from ..._auxlib._tgs_fit import _tgs_calc_effective_length_deriv


def tgs_calc_allmdc_group_nll(
    *,
    log_gain_rate: float,
    rate: float,
    shift: float,
    group_data: typing.Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_growth_exponent: float,
) -> typing.Tuple[float, float, float]:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns negative log likelihood (up to a constant) of one clade-size
    group's partitions, given as (branch counts, gain counts, mean
    lengths), and its derivatives with respect to log gain rate and to
    `rate` (selection minus loss rate). A branch of length `t` gains the
    trait with probability

        gain_rate * (exp(rate * t) - 1) / rate * exp(rate * shift)

    with exponents capped at `max_growth_exponent`.
    """
    num_trials, num_successes, lengths = group_data
    gain_rate = np.exp(log_gain_rate)
    effective_length = lengths * special.exprel(
        np.minimum(rate * lengths, max_growth_exponent)
    )
    effective_length_deriv = _tgs_calc_effective_length_deriv(
        pendant_length=lengths,
        rate=rate,
        max_growth_exponent=max_growth_exponent,
    )
    # survival over the shift, i.e., the time a clade of this size takes to
    # grow out to its sampled leaves
    shift_factor = np.exp(min(rate * shift, max_growth_exponent))
    p = gain_rate * effective_length * shift_factor
    nll, dnll_dp = calc_binomial_nll_proxy(num_trials, num_successes, p)
    dnll_dlog_gain = np.sum(dnll_dp * p)
    dnll_drate = np.sum(
        dnll_dp
        * gain_rate
        * shift_factor
        * (effective_length_deriv + shift * effective_length)
    )
    return nll, dnll_dlog_gain, dnll_drate
