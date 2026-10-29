import typing
import warnings

import numpy as np
from scipy import optimize, special

from ._calc_binomial_nll_proxy import calc_binomial_nll_proxy
from ._warn_experimental import warn_experimental


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


def _tgs_calc_effective_length_deriv(
    *, pendant_length: np.ndarray, rate: float, max_exponent: float
) -> np.ndarray:
    """Implementation detail for `_tgs_calc_nll`.

    Here, tgs stands for trait gain/selection.

    Returns the derivative, with respect to `rate`, of effective length
    (exp(rate * t) - 1) / rate for each pendant length t, with rate * t
    capped at `max_exponent`.

    The exact form, (t * exp(rate * t) - effective length) / rate, loses
    precision as rate * t approaches 0, as it divides a difference of nearly
    equal numbers by a small rate (relative error ~1e-16 / |rate * t|).
    In this regime, the Taylor series of the exponential function provides
    better numerical stability: with effective length = sum over j >= 1 of rate**(j - 1) * t**j / j!, the
    derivative is t**2 / 2 + rate * t**3 / 3 + O(rate**2), truncated with
    relative error ~(rate * t)**2 / 4; see
    https://en.wikipedia.org/wiki/Exponential_function#Power_series

    So, the series is used below a |rate * t| < 1e-6 cutoff and the exact
    form above it. At the cutoff, the exact form's error is ~1e-10 and the
    series' ~1e-13. Below it, the exact form degrades (and divides by zero
    at rate = 0), while the series improves.
    """
    cx = np.minimum(rate * pendant_length, max_exponent)
    effective_length = pendant_length * special.exprel(cx)
    is_near_zero_rate = np.abs(cx) < 1e-6
    series_deriv = pendant_length * pendant_length * (0.5 + cx / 3)
    exact_deriv = np.divide(
        pendant_length * np.exp(cx) - effective_length,
        rate,
        out=np.zeros_like(cx),
        where=~is_near_zero_rate,
    )
    return np.where(is_near_zero_rate, series_deriv, exact_deriv)


def _tgs_calc_nll(
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    pendant_length: np.ndarray,
    gain_rate: float,
    rate: float,
    *,
    max_exponent: float,
) -> typing.Tuple[float, float, float]:
    """Implementation detail for `tgs_fit`.

    Here, tgs stands for trait gain/selection.

    Returns negative log likelihood and its derivatives with respect to log
    gain rate and to `rate` (selection minus loss rate), with rate *
    pendant length capped at `max_exponent`.
    """
    # p = gain_rate * effective_length, where for pendant length t,
    # effective_length = (exp(rate * t) - 1) / rate is the integral of
    # exp(rate * s) over s in [0, t], i.e., pendant length weighted by how a
    # gain's carrier lineage grows (rate > 0) or shrinks (rate < 0) afterward;
    # exprel(x) = (exp(x) - 1) / x, finite at x = 0
    effective_length = pendant_length * special.exprel(
        np.minimum(rate * pendant_length, max_exponent)
    )
    effective_length_deriv = _tgs_calc_effective_length_deriv(
        pendant_length=pendant_length, rate=rate, max_exponent=max_exponent
    )

    p = gain_rate * effective_length
    nll, dnll_dp = calc_binomial_nll_proxy(num_trials, num_successes, p)
    # chain rule, as d p / d log(gain_rate) = p
    dnll_dlog_gain = np.sum(dnll_dp * p)
    dnll_drate = np.sum(dnll_dp * gain_rate * effective_length_deriv)
    return nll, dnll_dlog_gain, dnll_drate


def _tgs_fit_impl(
    *,
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    pendant_length: np.ndarray,
    loss_rate: float = 0.0,
    selection_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    max_exponent: float = 600.0,
) -> TgsFitResult:
    """Implementation detail for `tgs_fit`, without its experimental
    warning, for use within phyloframe.

    Here, tgs stands for trait gain/selection.
    """
    num_trials = np.asarray(num_trials, dtype=np.float64).ravel()
    num_successes = np.asarray(num_successes, dtype=np.float64).ravel()
    pendant_length = np.asarray(pendant_length, dtype=np.float64).ravel()

    keep = num_trials > 0
    n, k, x = num_trials[keep], num_successes[keep], pendant_length[keep]
    if not keep.any() or not num_successes.any():
        warnings.warn(
            "tgs_fit got no trait carriers, so gain rate is 0 and selection "
            "is undefined.",
        )
        return TgsFitResult(
            gain_rate=0.0, selection=np.nan, selection_lrt_naive=np.nan
        )

    # optimize selection in units of 1 / mean pendant length, so that both
    # parameters are of order one; otherwise, with long pendant lengths,
    # L-BFGS-B stops early on the tiny selection parameter
    unit = 1.0 / np.average(x, weights=n)

    def nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        log_gain, scaled_selection = params
        res, dlog_gain, drate = _tgs_calc_nll(
            n,
            k,
            x,
            np.exp(log_gain),
            scaled_selection * unit - loss_rate,
            max_exponent=max_exponent,
        )
        return res, np.array([dlog_gain, drate * unit])

    # under the null hypothesis of no selection (selection = 0), for the
    # likelihood-ratio test
    def null_nll(params: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        (log_gain,) = params
        res, dlog_gain, __ = _tgs_calc_nll(
            n, k, x, np.exp(log_gain), -loss_rate, max_exponent=max_exponent
        )
        return res, np.array([dlog_gain])

    minimize_kwargs = dict(
        jac=True,  # objective function returns its own gradient
        method="L-BFGS-B",  # gradient-based, for differentiable functions
    )
    # start log gain rate at its neutral estimate, carriers per unit pendant
    # length, and selection at each of `selection_starts`, keeping the best
    # fit in case the optimizer stalls in a local minimum
    log_gain_start = np.log(k.sum() / (n * x).sum())
    fits = [
        optimize.minimize(
            nll, [log_gain_start, selection_start], **minimize_kwargs
        )
        for selection_start in selection_starts
    ]
    # prefer converged fits
    converged = [fit for fit in fits if fit.success]
    best = min(converged or fits, key=lambda res: res.fun)  # .fun is nll
    null = optimize.minimize(  # fit gain rate alone, with selection = 0
        null_nll, best.x[:1], **minimize_kwargs
    )
    if not converged:
        warnings.warn(
            "tgs_fit optimizer did not converge from any starting point "
            f"({best.message}); estimates may be unreliable.",
        )
    if not null.success:
        warnings.warn(
            "tgs_fit optimizer did not converge under the null hypothesis "
            f"of no selection ({null.message}); likelihood-ratio statistic "
            "may be unreliable.",
        )
    return TgsFitResult(
        gain_rate=float(np.exp(best.x[0])),
        selection=float(best.x[1] * unit),
        # two-tailed, testing selection != 0 in either direction
        selection_lrt_naive=float(2 * (null.fun - best.fun)),
    )


@warn_experimental
def tgs_fit(
    *,
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    pendant_length: np.ndarray,
    loss_rate: float = 0.0,
    selection_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    max_exponent: float = 600.0,
) -> TgsFitResult:
    """Maximum likelihood fit of a binary trait's gain rate and net
    selection from the fraction of leaves carrying it as a function of
    pendant branch length.

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
        Fixed a priori trait loss rate.
    selection_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Starting values for selection, in units of 1 / mean pendant length;
        the best of the resulting fits is kept. Defaults span no cost, a
        mild cost, and a strong cost, under which carrier frequency
        saturates within about half a mean pendant length.
    max_exponent : float, default 600.0
        Cap on (selection - loss rate) * pendant length, so that optimizer
        trial steps far into positive selection stay finite (`p` is then
        clipped, zeroing its gradient), rather than overflowing to inf and
        giving nan gradients. Must be below ~709.78, where float64 exp
        overflows; the default leaves headroom for the further
        multiplication by pendant length and gain rate. Fits that reach the
        cap, i.e., whose carrier frequency would grow by a factor of ~e**600
        along a pendant branch, are not meaningful anyway.

    Returns
    -------
    TgsFitResult
        Named tuple of `gain_rate`, the estimated gain rate; `selection`,
        the estimated selection (negative for a cost); and
        `selection_lrt_naive`, the likelihood-ratio test (LRT) statistic for
        selection = 0.

        The likelihood-ratio statistic treats bins as independent, so it
        naively overstates significance when lineages are correlated (e.g.,
        through shared ancestry or space), and ignores uncertainty in any
        calibration of `pendant_length`. Prefer bootstrap standard errors
        for inference (see `num_bootstrap` in
        `alifestd_estimate_trait_gain_selection_pendant_asexual`).

    See Also
    --------
    alifestd_estimate_trait_gain_selection_pendant_asexual :
        Applies this fit to leaf pendant branches of a phylogeny, with
        bootstrap standard errors and optional calibration against neutral
        traits.

    References
    ----------
    Maddison, W. P., Midford, P. E., and Otto, S. P. (2007). Estimating a
    binary character's effect on speciation and extinction. Systematic
    Biology, 56(5), 701-710.
    https://doi.org/10.1080/10635150701607033
    """
    return _tgs_fit_impl(
        num_trials=num_trials,
        num_successes=num_successes,
        pendant_length=pendant_length,
        loss_rate=loss_rate,
        selection_starts=selection_starts,
        max_exponent=max_exponent,
    )
