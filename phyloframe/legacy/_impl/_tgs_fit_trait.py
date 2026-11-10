import logging
import typing
import warnings

import numpy as np

from ._tgs_minimize_nll import tgs_minimize_nll, tgs_pick_selection_rescale


def tgs_fit_trait(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
) -> typing.Dict[str, float]:
    """Implementation detail for `tgs_fit_traits_coarsegrain`.

    Maximum likelihood fit of one trait's gain rate and selection
    coefficient to its coarsened exposure episodes (see `tgs_calc_nll`),
    keeping the best fit from each of `selection_coef_starts`. Returns
    `gain_rate`, `selection_coef`, and their negative log likelihood,
    `nll`; without gains, zero gain rate and nan otherwise.

    Notes
    -----
    In BiSSE terms (Maddison et al. 2007), `gain_rate` corresponds to the
    0 -> 1 transition rate q01, `loss_rate` to q10, and `selection_coef` to
    the difference in net diversification rate between states, (lambda1 -
    mu1) - (lambda0 - mu0). Unlike BiSSE, which fits a full tree likelihood,
    this model considers each episode independently and assumes the trait
    is rare, so that it scales to large trees.

    References
    ----------
    Maddison, W. P., Midford, P. E., and Otto, S. P. (2007). Estimating a
    binary character's effect on speciation and extinction. Systematic
    Biology, 56(5), 701-710.
    https://doi.org/10.1080/10635150701607033
    """
    coarsened_df = nll_kwargs["coarsened_df"]
    num_gains = coarsened_df["num_successes"].sum()
    if not num_gains:
        return dict(gain_rate=0.0, selection_coef=np.nan, nll=np.nan)

    exposure = coarsened_df["num_trials"] * coarsened_df["mean_episode_length"]
    log_gain_start = np.log(num_gains / exposure.sum())  # no selection
    best = tgs_minimize_nll(
        nll_kwargs=nll_kwargs,
        starts=[(log_gain_start, start) for start in selection_coef_starts],
        fit_gain=True,
        fit_selection=True,
        loss_rate=loss_rate,
        growth_exponent_bound=growth_exponent_bound,
    )

    if not best.num_converged:
        warnings.warn(
            "tgs fit optimizer did not converge from any starting point "
            f"({best.message}); estimates may be unreliable.",
        )
    logging.debug(
        f"- tgs fit: {best.num_converged} of {len(selection_coef_starts)} "
        f"starts converged; best negative log likelihood {best.fun:.6g}",
    )
    log_gain_rate, selection = best.x
    return dict(
        gain_rate=float(np.exp(log_gain_rate)),
        selection_coef=float(
            selection * tgs_pick_selection_rescale(coarsened_df)
        ),
        nll=float(best.fun),
    )
