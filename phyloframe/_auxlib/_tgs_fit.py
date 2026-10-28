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
        lineage's expected number of sampled descendants grows relative to
        a non-carrier's (negative for a cost); nan if no leaf carries the
        trait.
    selection_lrt_naive : float
        Likelihood-ratio test (LRT) statistic for selection = 0, naively
        treating bins as independent; nan if no leaf carries the trait.
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
    # p = gain_rate * effective_length, where for pendant length t,
    # effective_length = (exp(rate * t) - 1) / rate is the integral of
    # exp(rate * s) over s in [0, t], i.e., pendant length weighted by how a
    # gain's carrier lineage grows (rate > 0) or shrinks (rate < 0) afterward
    cx = rate * pendant_length
    # exprel(x) = (exp(x) - 1) / x, finite at x = 0
    effective_length = pendant_length * special.exprel(cx)

    # d effective_length / d rate, from a series expansion near rate = 0,
    # where the exact form divides by zero
    is_near_zero_rate = np.abs(cx) < 1e-6
    series_deriv = pendant_length * pendant_length * (0.5 + cx / 3)
    exact_deriv = np.divide(
        pendant_length * np.exp(cx) - effective_length,
        rate,
        out=np.zeros_like(cx),
        where=~is_near_zero_rate,
    )
    effective_length_deriv = np.where(
        is_near_zero_rate, series_deriv, exact_deriv
    )

    p = gain_rate * effective_length
    nll, dnll_dp = calc_binomial_nll(num_trials, num_successes, p)
    return (
        nll,
        np.sum(dnll_dp * p),
        np.sum(dnll_dp * gain_rate * effective_length_deriv),
    )


def tgs_fit(
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    pendant_length: np.ndarray,
    *,
    loss_rate: float = 0.0,
    selection_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
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

    i.e., gains arrive at constant `gain_rate`, after which a carrier
    lineage's expected number of sampled descendants, relative to a
    non-carrier's, changes at rate `c`. At `c` = 0, `p` grows linearly in
    `t`; under a cost (`c` < 0), it saturates at ~`gain_rate` / |`c`|.

    In BiSSE terms (Maddison et al. 2007), `gain_rate` corresponds to the
    0 -> 1 transition rate q01, `loss_rate` to q10, and `selection` to the
    difference in net diversification rate between states, (lambda1 - mu1)
    - (lambda0 - mu0). Unlike BiSSE, which fits a full tree likelihood,
    this model considers each pendant branch independently and assumes the
    trait is rare, so that `p` is proportional to `gain_rate` (ignoring,
    e.g., a lineage gaining the trait more than once); this lets it scale
    to large trees with rare traits.

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
    selection_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Starting values for selection, in units of 1 / mean pendant length;
        the best of the resulting fits is kept. Defaults span no cost, a
        mild cost, and a strong cost, under which carrier frequency
        saturates within about half a mean pendant length.

    Returns
    -------
    TgsFitResult
        Named tuple of `gain_rate`, the estimated gain rate; `selection`,
        the estimated selection (negative for a cost); and
        `selection_lrt_naive`, the likelihood-ratio test (LRT) statistic for
        selection = 0.

        The likelihood-ratio statistic treats bins as independent, so it
        naively overstates significance when lineages are correlated (e.g.,
        through shared ancestry or space). Prefer a cluster bootstrap for
        inference, which resamples groups of correlated leaves together
        (see `cluster_ids` and `num_bootstrap` in
        `alifestd_estimate_trait_gain_selection_asexual`).

    See Also
    --------
    alifestd_estimate_trait_gain_selection_asexual :
        Applies this fit to leaf pendant branches of a phylogeny.

    References
    ----------
    Maddison, W. P., Midford, P. E., and Otto, S. P. (2007). Estimating a
    binary character's effect on speciation and extinction. Systematic
    Biology, 56(5), 701-710.
    https://doi.org/10.1080/10635150701607033
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

    # under the null hypothesis of no selection (selection = 0), for the
    # likelihood-ratio test
    def null_nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        res, dlog_gain, __ = _tgs_calc_nll(
            n, k, x, np.exp(params[0]), -loss_rate
        )
        return res, np.array([dlog_gain])

    # L-BFGS-B minimizes a differentiable nonlinear function using its
    # gradient; jac=True as nll returns its own gradient
    minimize_kwargs = dict(jac=True, method="L-BFGS-B")
    # start log gain rate at its neutral estimate, carriers per unit pendant
    # length, and selection at each of `selection_starts`, keeping the best
    # fit in case the optimizer stalls in a local minimum
    log_gain0 = np.log(k.sum() / (n * x).sum())
    best = min(
        (
            optimize.minimize(nll, [log_gain0, selection0], **minimize_kwargs)
            for selection0 in selection_starts
        ),
        key=lambda res: res.fun,  # .fun is the minimized nll
    )
    null = optimize.minimize(  # fit gain rate alone, with selection = 0
        null_nll, best.x[:1], **minimize_kwargs
    )
    return TgsFitResult(
        gain_rate=float(np.exp(best.x[0])),
        selection=float(best.x[1] * unit),
        selection_lrt_naive=float(2 * (null.fun - best.fun)),
    )
