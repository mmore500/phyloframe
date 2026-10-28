import typing

import numpy as np
from scipy import special


def calc_binomial_nll(
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    p: np.ndarray,
) -> typing.Tuple[float, np.ndarray]:
    """Calculate binomial negative log likelihood, up to a constant, and its
    gradient with respect to success probability.

    Omits the log binomial coefficient, which does not depend on `p`, so
    that repeated evaluation (e.g., within an optimizer) is several times
    faster than `scipy.stats.binom.logpmf`.

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

    nll = -np.sum(
        special.xlogy(num_successes, clipped_p)
        + special.xlog1py(num_failures, -clipped_p)
    )
    dnll_dp = np.where(
        clipped_p != raw_p,
        0.0,
        num_failures / (1 - clipped_p) - num_successes / clipped_p,
    )
    return float(nll), dnll_dp
