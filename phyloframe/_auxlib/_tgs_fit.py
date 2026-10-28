import typing

import numpy as np
from scipy import optimize, special

from ._calc_binomial_nll import calc_binomial_nll


class TgsFitResult(typing.NamedTuple):
    """Result of `tgs_fit`.

    Here, tgs stands for trait gain/selection.

    Attributes
    ----------
    gain_rate : float
        Estimated rate at which non-carrier lineages gain the trait, per
        unit pendant length; 0.0 if no leaf carries the trait.
    selection : float
        Estimated net selection on the trait, the rate at which a carrier
        lineage's weight changes relative to a non-carrier's (negative for a
        cost); nan if no leaf carries the trait.
    selection_lrt_naive : float
        Likelihood-ratio statistic for selection = 0, treating bins as
        independent; nan if no leaf carries the trait.
    """

    gain_rate: float
    selection: float
    selection_lrt_naive: float


def _tgs_calc_nll(
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    pendant_length: np.ndarray,
    gain_rate: float,
    rate: float,
) -> typing.Tuple[float, float, float]:
    """Implementation detail for `tgs_fit`.

    Here, tgs stands for trait gain/selection.

    Returns negative log likelihood and its derivatives with respect to log
    gain rate and to `rate` (selection minus loss rate).
    """
    # p = gain_rate * g, with g = (exp(rate * t) - 1) / rate for pendant
    # length t, and dg = d g / d rate
    cx = rate * pendant_length
    g = pendant_length * special.exprel(cx)  # exprel(x) = (exp(x) - 1) / x
    small = np.abs(cx) < 1e-6  # use series expansion of dg near rate = 0
    dg = np.where(
        small,
        pendant_length * pendant_length * (0.5 + cx / 3),
        (pendant_length * np.exp(cx) - g) / np.where(small, 1.0, rate),
    )
    p = gain_rate * g
    nll, dnll_dp = calc_binomial_nll(num_trials, num_successes, p)
    return nll, np.sum(dnll_dp * p), np.sum(dnll_dp * gain_rate * dg)


def tgs_fit(
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    pendant_length: np.ndarray,
    *,
    loss_rate: float = 0.0,
) -> TgsFitResult:
    """Maximum likelihood fit of a binary trait's gain rate and net
    selection from how often leaves carry it, given their pendant branch
    lengths.

    Here, tgs stands for trait gain/selection, marking utilities specific to
    this estimator.

    A leaf's pendant branch starts without the trait, so pendant length is
    an exposure window: a leaf with pendant length `t` carries the trait
    with probability

        p = gain_rate * (exp(c * t) - 1) / c,  c = selection - loss_rate

    i.e., gains arrive at constant `gain_rate` and each carrier lineage's
    weight then changes at rate `c`. At `c` = 0, `p` grows linearly in
    `t`; under a cost (`c` < 0), it saturates at ~`gain_rate` / |`c`|.

    In BiSSE terms (Maddison et al. 2007), `gain_rate` corresponds to the
    0 -> 1 transition rate q01, `loss_rate` to q10, and `selection` to the
    difference in net diversification rate between states, (lambda1 - mu1)
    - (lambda0 - mu0). Unlike BiSSE, which fits a full tree likelihood,
    this model considers each pendant branch independently and
    to first order in the trait frequency, so it scales to large trees with
    rare traits.

    Parameters
    ----------
    num_trials : np.ndarray
        Number of leaves in each bin.
    num_successes : np.ndarray
        Number of leaves carrying the trait in each bin.
    pendant_length : np.ndarray
        Mean pendant branch length of each bin's leaves, optionally rescaled
        to effective exposure (e.g., by calibration against neutral
        traits).
    loss_rate : float, default 0.0
        Known trait loss rate; with few carriers, loss and cost are
        confounded, so loss is fixed rather than fit.

    Returns
    -------
    TgsFitResult
        Named tuple of `gain_rate`, the estimated gain rate; `selection`,
        the estimated selection (negative for a cost); and
        `selection_lrt_naive`, the likelihood-ratio statistic for
        selection = 0.

        The likelihood-ratio statistic treats bins as independent, so it
        overstates significance when lineages are correlated (e.g., through
        shared ancestry or space); prefer a block bootstrap for inference.

    See Also
    --------
    alifestd_estimate_trait_gain_selection_asexual :
        Applies this fit to leaf pendant branches of a phylogeny.

    References
    ----------
    Maddison, W. P., Midford, P. E., and Otto, S. P. (2007). Estimating a
    binary character's effect on speciation and extinction. Systematic
    Biology, 56(5), 701-710.
    """
    num_trials = np.asarray(num_trials, dtype=np.float64).ravel()
    num_successes = np.asarray(num_successes, dtype=np.float64).ravel()
    pendant_length = np.asarray(pendant_length, dtype=np.float64).ravel()

    keep = num_trials > 0
    n, k, x = num_trials[keep], num_successes[keep], pendant_length[keep]
    if not keep.any() or not num_successes.any():
        return TgsFitResult(0.0, np.nan, np.nan)

    # optimize selection in units of 1 / mean pendant length, for conditioning
    unit = 1.0 / np.average(x, weights=n)

    def nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        res, dlog_gain, drate = _tgs_calc_nll(
            n, k, x, np.exp(params[0]), params[1] * unit - loss_rate
        )
        return res, np.array([dlog_gain, drate * unit])

    def null_nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        res, dlog_gain, __ = _tgs_calc_nll(
            n, k, x, np.exp(params[0]), -loss_rate
        )
        return res, np.array([dlog_gain])

    # L-BFGS-B minimizes a differentiable nonlinear function (optionally
    # under box constraints); jac=True as nll returns its own gradient
    minimize_kwargs = dict(
        jac=True,
        method="L-BFGS-B",
        options=dict(ftol=1e-15, gtol=1e-10, maxiter=1000),
    )
    # start log gain rate at its neutral estimate, carriers per unit pendant
    # length; start selection, in units of 1 / mean pendant length, at no
    # cost, a mild cost, and a strong cost, keeping the best fit in case the
    # optimizer stalls in a local minimum
    log_gain0 = np.log(k.sum() / (n * x).sum())
    best = min(
        (
            optimize.minimize(nll, [log_gain0, selection0], **minimize_kwargs)
            for selection0 in (-0.3, 0.0, -1.8)
        ),
        key=lambda res: res.fun,  # .fun is the minimized nll
    )
    null = optimize.minimize(null_nll, best.x[:1], **minimize_kwargs)
    return TgsFitResult(
        gain_rate=float(np.exp(best.x[0])),
        selection=float(best.x[1] * unit),
        selection_lrt_naive=float(2 * (null.fun - best.fun)),
    )
