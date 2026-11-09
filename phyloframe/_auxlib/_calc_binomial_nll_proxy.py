import typing

import numpy as np
from scipy import special


def calc_binomial_nll_proxy(
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    p: np.ndarray,
) -> typing.Tuple[float, np.ndarray]:
    """Calculate a proxy for binomial negative log likelihood (up to an
    additive constant), and its gradient with respect to success
    probability.

    Omits log binomial coefficients, which do not depend on `p`, for speedup
    over `scipy.stats.binom.logpmf` under repeated evaluation (e.g., within
    an optimizer). Differences between results for different `p` and the
    gradient are exact.

    Parameters
    ----------
    num_trials : np.ndarray
        Number of trials in each bin.
    num_successes : np.ndarray
        Number of successes in each bin.
    p : np.ndarray
        Success probability of each bin.

        Values outside the open interval (0, 1) are clipped to keep the
        result finite, and their gradient is zeroed.

    Returns
    -------
    typing.Tuple[float, np.ndarray]
        Negative log likelihood summed over bins, and its derivative with
        respect to each bin's `p`.

    See Also
    --------
    scipy.stats.binom.logpmf :
        Normalized binomial log probability mass function.
    """
    finfo = np.finfo(np.float64)
    raw_p = np.asarray(p, dtype=np.float64)
    clipped_p = np.clip(raw_p, finfo.tiny, 1 - finfo.epsneg)
    num_successes = np.asarray(num_successes, dtype=np.float64)
    num_failures = np.asarray(num_trials, dtype=np.float64) - num_successes
    # optimization: take log(p) and 1 / p only for nonzero successes
    has_successes = num_successes > 0
    successes = num_successes[has_successes]
    success_p = clipped_p[has_successes]

    # for n trials and k successes, up to the additive constant -log C(n, k),
    #   nll = -sum(k * log(p) + (n - k) * log(1 - p))
    #   d nll / d p = (n - k) / (1 - p) - k / p
    nll = -(
        np.sum(successes * np.log(success_p))
        + np.sum(special.xlog1py(num_failures, -clipped_p))
    )
    dnll_dp = np.asarray(num_failures / (1 - clipped_p))  # scalar ok
    dnll_dp[has_successes] -= successes / success_p
    dnll_dp[clipped_p != raw_p] = 0.0
    return nll, dnll_dp
