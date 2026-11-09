import logging
import typing
import warnings

import numpy as np

from ._tgs_minimize_nll import (
    tgs_minimize_nll,
    tgs_pick_log_gain_rate_start,
    tgs_pick_selection_rescale,
)


def tgs_fit_trait(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
) -> typing.Dict[str, float]:
    """Implementation detail for `alifestd_tgs_estimate_polars`, via
    `tgs_fit_coarsened_traits`.
    Acronym tgs stands for trait gain/selection.

    Maximum likelihood fit of one binary trait's gain rate and selection
    coefficient to its coarsened exposure episodes,
    `nll_kwargs["coarsened_df"]` (see `_tgs_coarsen_episodes`), minimizing
    `tgs_calc_nll(**nll_kwargs, ...)` from each of `selection_coef_starts`
    (in units of 1 / mean branch length) and keeping the best fit. Without
    gains, skips fitting, giving zero gain rate and undefined (nan)
    selection coefficient. Returns `gain_rate` and `selection_coef`.

    An exposure episode is a branch starting without the trait, so branch
    length is an exposure window: an episode of length `t` gains the trait
    with probability

        p = gain_rate * (exp(c * t) - 1) / c,  c = selection_coef - loss_rate

    i.e., gains arrive at constant `gain_rate`, after which a carrier
    lineage's expected number of sampled descendants, relative to a
    non-carrier's, changes at rate `c`. At `c` = 0, `p` grows linearly in
    `t`; under a cost (`c` < 0), it saturates at ~`gain_rate` / |`c`|. For
    inner branches, `p` also has a survival factor; see `tgs_calc_nll`.

    In BiSSE terms (Maddison et al. 2007), `gain_rate` corresponds to the
    0 -> 1 transition rate q01, `loss_rate` to q10, and `selection_coef` to
    the difference in net diversification rate between states, (lambda1 -
    mu1) - (lambda0 - mu0). Unlike BiSSE, which fits a full tree likelihood,
    this model considers each episode independently and assumes the trait
    is rare, so that `p` is proportional to `gain_rate` (ignoring, e.g., a
    lineage gaining the trait more than once); this lets it scale to large
    trees with rare traits.

    References
    ----------
    Maddison, W. P., Midford, P. E., and Otto, S. P. (2007). Estimating a
    binary character's effect on speciation and extinction. Systematic
    Biology, 56(5), 701-710.
    https://doi.org/10.1080/10635150701607033
    """
    coarsened_df = nll_kwargs["coarsened_df"]
    if not coarsened_df["num_successes"].to_numpy().any():
        return dict(gain_rate=0.0, selection_coef=np.nan)

    selection_rescale = tgs_pick_selection_rescale(coarsened_df)
    if not np.isfinite(selection_rescale):  # no calibrated exposure
        raise ValueError(
            "a trait has gains but no calibrated exposure, as its exposure "
            "episodes all fall in calibration neighborhoods without neutral "
            "trait gains; use coarser calibration, e.g., fewer, larger "
            "calibration bins (see alifestd_mark_joint_quantile_bins_polars) "
            "or larger calibrate_neighborhood_fractions",
        )
    log_gain_start = tgs_pick_log_gain_rate_start(coarsened_df)
    best = tgs_minimize_nll(
        nll_kwargs=nll_kwargs,
        starts=[(log_gain_start, start) for start in selection_coef_starts],
        fit_gain=True,
        fit_selection=True,
        selection_rescale=selection_rescale,
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
        selection_coef=float(best.x[1] * selection_rescale),
    )
