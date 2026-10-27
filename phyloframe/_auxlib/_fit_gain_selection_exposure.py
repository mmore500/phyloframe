import functools
import typing

import numpy as np
from scipy import optimize


def _calc_gain_selection_exposure_nll(
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    exposure: np.ndarray,
    gain_rate: float,
    rate: float,
) -> typing.Tuple[float, float, float]:
    """Implementation detail for `fit_gain_selection_exposure`.

    Returns negative log likelihood and its derivatives with respect to log
    gain rate and to `rate` (selection minus loss rate).
    """
    cx = rate * exposure
    small = np.abs(cx) < 1e-6  # use series expansion near rate = 0
    safe_rate = np.where(small, 1.0, rate)
    g = np.where(small, exposure * (1 + cx / 2), np.expm1(cx) / safe_rate)
    dg = np.where(
        small,
        exposure * exposure * (0.5 + cx / 3),
        (exposure * np.exp(cx) - g) / safe_rate,
    )
    p = gain_rate * g
    is_clipped = (p <= 1e-300) | (p >= 1 - 1e-12)
    p = np.clip(p, 1e-300, 1 - 1e-12)

    nll = -np.sum(
        num_successes * np.log(p) + (num_trials - num_successes) * np.log1p(-p)
    )
    dnll_dp = np.where(
        is_clipped,
        0.0,
        (num_trials - num_successes) / (1 - p) - num_successes / p,
    )
    return nll, np.sum(dnll_dp * p), np.sum(dnll_dp * gain_rate * dg)


def fit_gain_selection_exposure(
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    exposure: np.ndarray,
    *,
    loss_rate: float = 0.0,
) -> typing.Tuple[float, float, float]:
    """Maximum likelihood fit of trait gain rate and net selection from
    binomial counts observed at known exposures.

    A lineage observed after `exposure` time units, starting without the
    trait, carries the trait with probability

        p = gain_rate * (exp(c * exposure) - 1) / c,  c = selection - loss_rate

    i.e., gains arrive at constant `gain_rate` and each carrier lineage's
    weight then changes at rate `c`. At `c` = 0, `p` grows linearly in
    exposure; under a cost (`c` < 0), it saturates at ~`gain_rate` / |`c`|.

    Parameters
    ----------
    num_trials : np.ndarray
        Number of observed lineages in each bin.
    num_successes : np.ndarray
        Number of observed lineages carrying the trait in each bin.
    exposure : np.ndarray
        Mean exposure (e.g., pendant branch length) of each bin.
    loss_rate : float, default 0.0
        Known trait loss rate; with few carriers, loss and cost are
        confounded, so loss is fixed rather than fit.

    Returns
    -------
    typing.Tuple[float, float, float]
        Estimated gain rate, estimated selection (negative for a cost), and
        likelihood-ratio statistic for selection = 0.

        The likelihood-ratio statistic treats bins as independent, so it
        overstates significance when lineages are correlated (e.g., through
        shared ancestry or space); prefer a block bootstrap for inference.
    """
    num_trials = np.asarray(num_trials, dtype=np.float64).ravel()
    num_successes = np.asarray(num_successes, dtype=np.float64).ravel()
    exposure = np.asarray(exposure, dtype=np.float64).ravel()

    keep = num_trials > 0
    n, k, x = num_trials[keep], num_successes[keep], exposure[keep]
    if not len(n) or not k.sum():
        return 0.0, np.nan, np.nan

    # optimize selection in units of 1 / mean exposure, for conditioning
    unit = 1.0 / np.average(x, weights=n)

    def nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        res, dlog_gain, drate = _calc_gain_selection_exposure_nll(
            n, k, x, np.exp(params[0]), params[1] * unit - loss_rate
        )
        return res, np.array([dlog_gain, drate * unit])

    def null_nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        res, dlog_gain, __ = _calc_gain_selection_exposure_nll(
            n, k, x, np.exp(params[0]), -loss_rate
        )
        return res, np.array([dlog_gain])

    minimize = functools.partial(
        optimize.minimize,
        jac=True,
        method="L-BFGS-B",
        options=dict(ftol=1e-15, gtol=1e-10, maxiter=1000),
    )
    x0 = np.array([np.log(k.sum() / (n * x).sum()), -0.3])
    best = min(
        (
            minimize(nll, start)
            for start in (x0, x0 + [0.0, 0.3], x0 + [0.0, -1.5])
        ),
        key=lambda res: res.fun,
    )
    null = minimize(null_nll, best.x[:1])
    return (
        float(np.exp(best.x[0])),
        float(best.x[1] * unit),
        float(2 * (null.fun - best.fun)),
    )
