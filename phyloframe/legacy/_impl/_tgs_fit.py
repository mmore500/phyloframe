import functools
import logging
import typing
import warnings

import numpy as np
import pandas as pd
from scipy import optimize

from ._tgs_calc_nll import tgs_calc_nll


class TgsFitResult(typing.NamedTuple):
    """Result of `tgs_fit`. Acronym tgs stands for trait gain/selection.

    Attributes
    ----------
    gain_rate : float
        Estimated rate at which non-carrier lineages gain the trait, per
        unit pendant length; 0.0 if no leaf carries the trait.
    selection_coef : float
        Estimated selection coefficient of the trait, the rate at which a
        carrier lineage's expected number of sampled descendants grows relative
        to a non-carrier's (negative for a cost); nan if no leaf carries the
        trait.
    selection_coef_lrt_naive : float
        Likelihood-ratio test (LRT) statistic for selection coefficient = 0,
        naively treating bins as independent; nan if no leaf carries the trait.
    """

    gain_rate: float
    selection_coef: float
    selection_coef_lrt_naive: float


def _tgs_calc_scales(partitions: pd.DataFrame) -> typing.Tuple[float, float]:
    """Implementation detail for `tgs_fit_partitions` and
    `tgs_calc_gain_ratio_lrts`. Acronym tgs stands for trait gain/selection.

    Returns the selection scale, 1 / mean branch length, and the neutral
    log gain rate, log(gains per unit branch length), of `partitions`.
    """
    num_trials = partitions["num_trials"].to_numpy()
    exposure = np.sum(num_trials * partitions["length"].to_numpy())
    return (
        num_trials.sum() / exposure,
        np.log(partitions["num_successes"].sum() / exposure),
    )


def tgs_minimize_nll(
    *,
    calc_nll: typing.Callable[..., typing.Tuple[float, float, float]],
    starts: typing.Sequence[typing.Tuple[float, float]],
    fit_gain: bool,
    fit_selection: bool,
    selection_scale: float,
    loss_rate: float,
) -> optimize.OptimizeResult:
    """Implementation detail for `tgs_fit_partitions` and
    `tgs_calc_gain_ratio_lrts`. Acronym tgs stands for trait gain/selection.

    Minimizes `calc_nll(log_gain_rate=..., rate=...)` (see `tgs_calc_nll`) by
    L-BFGS-B over log gain rate if `fit_gain` and selection coefficient if
    `fit_selection`, holding the others at their `starts` values. Starts from
    each of `starts` and returns the best fit, preferring converged ones, with
    both parameters in `.x` and the number of converged fits in
    `.num_converged`.

    The selection coefficient is optimized, and given in `starts`, as a
    multiple of `selection_scale`, a typical rate, e.g., 1 / mean branch
    length, so that both parameters are of order one; otherwise, with long
    lengths, L-BFGS-B stops early on the tiny selection coefficient.
    """
    is_free = np.array([fit_gain, fit_selection])

    def nll(
        free_params: np.ndarray, start: np.ndarray
    ) -> typing.Tuple[float, np.ndarray]:
        params = start.copy()
        params[is_free] = free_params
        log_gain_rate, scaled_selection = params
        res, dlog_gain, drate = calc_nll(
            log_gain_rate=log_gain_rate,
            rate=scaled_selection * selection_scale - loss_rate,
        )
        return res, np.array([dlog_gain, drate * selection_scale])[is_free]

    fits = []
    for start in starts:
        start = np.asarray(start, dtype=np.float64)
        fit = optimize.minimize(
            nll,
            start[is_free],
            args=(start,),
            jac=True,  # objective function returns its own gradient
            method="L-BFGS-B",  # gradient-based, for differentiable functions
        )
        params = start.copy()
        params[is_free] = fit.x
        fit.x = params
        fits.append(fit)
    converged = [fit for fit in fits if fit.success]
    best = min(converged or fits, key=lambda fit: fit.fun)  # .fun is nll
    best.num_converged = len(converged)
    return best


def tgs_fit_partitions(
    *,
    calc_nll: typing.Callable[..., typing.Tuple[float, float, float]],
    partitions: pd.DataFrame,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
) -> TgsFitResult:
    """Implementation detail for `tgs_fit` and `tgs_fit_trait_partitions`.
    Acronym tgs stands for trait gain/selection.

    Fits one trait's gain rate and selection coefficient coefficient to its
    `partitions` (with columns `num_trials`, `num_successes`, and mean branch
    `length`), minimizing `calc_nll(log_gain_rate=..., rate=...)` (see
    `tgs_calc_nll`) from each of `selection_coef_starts` (in units of 1 / mean
    branch length) and keeping the best fit, then fitting gain rate alone under
    no selection for the likelihood-ratio statistic. Without gains, skips
    fitting, giving zero gain rate and undefined selection coefficient.
    """
    if not partitions["num_successes"].to_numpy().any():
        return TgsFitResult(
            gain_rate=0.0,
            selection_coef=np.nan,
            selection_coef_lrt_naive=np.nan,
        )

    selection_scale, log_gain_start = _tgs_calc_scales(partitions)
    minimize = functools.partial(
        tgs_minimize_nll,
        calc_nll=calc_nll,
        selection_scale=selection_scale,
        loss_rate=loss_rate,
    )
    best = minimize(
        starts=[(log_gain_start, start) for start in selection_coef_starts],
        fit_gain=True,
        fit_selection=True,
    )
    # under the null hypothesis of no selection, for the likelihood-ratio test
    null = minimize(
        starts=[(best.x[0], 0.0)], fit_gain=True, fit_selection=False
    )
    if not best.num_converged:
        warnings.warn(
            "tgs fit optimizer did not converge from any starting point "
            f"({best.message}); estimates may be unreliable.",
        )
    if not null.success:
        warnings.warn(
            "tgs fit optimizer did not converge under the null hypothesis "
            f"of no selection ({null.message}); likelihood-ratio statistic "
            "may be unreliable.",
        )
    logging.debug(
        f"- tgs fit: {best.num_converged} of "
        f"{len(selection_coef_starts)} starts "
        f"converged; best negative log likelihood {best.fun:.6g}, null "
        f"{null.fun:.6g}",
    )
    return TgsFitResult(
        gain_rate=float(np.exp(best.x[0])),
        selection_coef=float(best.x[1] * selection_scale),
        # two-tailed, testing selection coefficient != 0 in either direction
        selection_coef_lrt_naive=float(2 * (null.fun - best.fun)),
    )


def tgs_fit(
    *,
    num_trials: np.ndarray,
    num_successes: np.ndarray,
    pendant_length: np.ndarray,
    loss_rate: float = 0.0,
    selection_coef_starts: typing.Sequence[float] = (0.0, -0.3, -1.8),
    growth_exponent_bound: float = 600.0,
) -> TgsFitResult:
    """Maximum likelihood fit of a binary trait's gain rate and selection
    coefficient from the fraction of leaves carrying it as a function of
    pendant branch length.

    Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars`.
    Acronym tgs stands for trait gain/selection.

    A leaf's pendant branch starts without the trait, so pendant length is
    an exposure window: a leaf with pendant length `t` carries the trait
    with probability

        p = gain_rate * (exp(c * t) - 1) / c,  c = selection_coef - loss_rate

    i.e., gains arrive at constant `gain_rate`, after which a carrier
    lineage's expected number of sampled descendants, relative to a
    non-carrier's, changes at rate `c`. At `c` = 0, `p` grows linearly in
    `t`; under a cost (`c` < 0), it saturates at ~`gain_rate` / |`c`|.

    In BiSSE terms (Maddison et al. 2007), `gain_rate` corresponds to the
    0 -> 1 transition rate q01, `loss_rate` to q10, and `selection_coef` to the
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
    selection_coef_starts : Sequence[float], default (0.0, -0.3, -1.8)
        Starting values for the selection coefficient, in units of 1 / mean
        pendant length; the best of the resulting fits is kept.
    growth_exponent_bound : float, default 600.0
        Bound on the carrier growth exponent, (selection coefficient - loss
        rate) * pendant length, keeping optimizer trial steps far into positive
        selection finite.

    Returns
    -------
    TgsFitResult
        Named tuple of `gain_rate`, the estimated gain rate; `selection_coef`,
        the estimated selection coefficient (negative for a cost); and
        `selection_coef_lrt_naive`, the likelihood-ratio test (LRT) statistic
        for selection coefficient = 0.

        The likelihood-ratio statistic treats bins as independent, so it
        naively overstates significance when lineages are correlated (e.g.,
        through shared ancestry or space), and ignores uncertainty in any
        calibration of `pendant_length`. Prefer bootstrap standard errors
        for inference (see `num_bootstrap` in
        `alifestd_tgs_pendantmdc_estimate_polars`).

    See Also
    --------
    alifestd_tgs_pendantmdc_estimate_polars :
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
    num_trials = np.asarray(num_trials, dtype=np.float64).ravel()
    num_successes = np.asarray(num_successes, dtype=np.float64).ravel()
    if not num_successes.any():
        warnings.warn(
            "tgs_fit got no trait carriers, so gain rate is 0 and the "
            "selection coefficient is undefined.",
        )
    partitions = pd.DataFrame(
        {
            "num_trials": num_trials,
            "num_successes": num_successes,
            "length": np.asarray(pendant_length, dtype=np.float64).ravel(),
        },
    )
    return tgs_fit_partitions(
        calc_nll=functools.partial(
            tgs_calc_nll,
            partitions=partitions,
            growth_exponent_bound=growth_exponent_bound,
        ),
        partitions=partitions,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
    )
