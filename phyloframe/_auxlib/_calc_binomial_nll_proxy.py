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

    The result is offset from the true negative log likelihood by the summed
    log binomial coefficients, which do not depend on `p`. So, differences
    between results for different `p` (e.g., likelihood-ratio statistics),
    the gradient, and the minimizing `p` are all exact.

    Avoids calculating log binomial coefficients, providing substantial
    speedup over `scipy.stats.binom.logpmf` for repeated evaluation (e.g.,
    within an optimizer).

    Parameters
    ----------
    num_trials : np.ndarray
        Number of trials in each bin.
    num_successes : np.ndarray
        Number of successes in each bin.
    p : np.ndarray
        Success probability of each bin.

        Values outside the open interval (0, 1), e.g., from underflow or
        optimizer trial steps overshooting, are clipped to keep the result
        finite, and their gradient is zeroed.

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
    num_failures = num_trials - num_successes

    # for n trials and k successes, up to the additive constant -log C(n, k),
    #   nll = -sum(k * log(p) + (n - k) * log(1 - p))
    #   d nll / d p = (n - k) / (1 - p) - k / p
    nll = -np.sum(
        special.xlogy(num_successes, clipped_p)
        + special.xlog1py(num_failures, -clipped_p)
    )
    dnll_dp = np.where(
        clipped_p != raw_p,
        0.0,
        num_failures / (1 - clipped_p) - num_successes / clipped_p,
    )
    return nll, dnll_dp
