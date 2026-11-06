import logging
import typing
import warnings

import numpy as np
import polars as pl

from ._tgs_calc_trait_selection_lrts import tgs_calc_trait_selection_lrt
from ._tgs_minimize_nll import _tgs_calc_scales, tgs_minimize_nll


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


def tgs_fit_trait(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
) -> typing.Dict[str, float]:
    """Implementation detail for `tgs_fit` and `tgs_fit_trait_partitions`.
    Acronym tgs stands for trait gain/selection.

    Fits one trait's gain rate and selection coefficient to its partitions,
    `nll_kwargs["partitions"]` (with columns `num_trials`, `num_successes`,
    and mean branch `length`), minimizing `tgs_calc_nll(**nll_kwargs, ...)`
    from each of `selection_coef_starts` (in units of 1 / mean branch length)
    and keeping the best fit. Without gains, skips fitting, giving zero gain
    rate and undefined selection coefficient. Returns `gain_rate` and
    `selection_coef`.
    """
    partitions = nll_kwargs["partitions"]
    if not partitions["num_successes"].to_numpy().any():
        return dict(gain_rate=0.0, selection_coef=np.nan)

    selection_scale, log_gain_start = _tgs_calc_scales(partitions)
    best = tgs_minimize_nll(
        nll_kwargs=nll_kwargs,
        starts=[(log_gain_start, start) for start in selection_coef_starts],
        fit_gain=True,
        fit_selection=True,
        selection_scale=selection_scale,
        loss_rate=loss_rate,
    )
    if not best.num_converged:
        warnings.warn(
            "tgs fit optimizer did not converge from any starting point "
            f"({best.message}); estimates may be unreliable.",
        )
    logging.debug(
        f"- tgs fit: {best.num_converged} of "
        f"{len(selection_coef_starts)} starts "
        f"converged; best negative log likelihood {best.fun:.6g}",
    )
    return dict(
        gain_rate=float(np.exp(best.x[0])),
        selection_coef=float(best.x[1] * selection_scale),
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
    nll_kwargs = dict(
        partitions=pl.DataFrame(
            {
                "num_trials": num_trials,
                "num_successes": num_successes,
                "length": np.asarray(pendant_length, dtype=np.float64).ravel(),
            },
        ),
        growth_exponent_bound=growth_exponent_bound,
    )
    fit = tgs_fit_trait(
        nll_kwargs=nll_kwargs,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
    )
    return TgsFitResult(
        **fit,
        selection_coef_lrt_naive=tgs_calc_trait_selection_lrt(
            nll_kwargs=nll_kwargs, loss_rate=loss_rate, **fit
        ),
    )
