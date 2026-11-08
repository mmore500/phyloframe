import typing
import warnings

import numpy as np
import polars as pl
from scipy import sparse

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy
from ._tgs_calc_calibration_rescaling import (
    tgs_calc_calibration_rescaling,
)


def tgs_calc_heldout_nll(
    *,
    calibration_neighborhoods: sparse.spmatrix,
    trait_bin_counts_df: pl.DataFrame,
    trait: int,
    trait_name: str,
    neutral_traits: typing.Sequence[int],
    max_rate_ratio: float,
) -> float:
    """Implementation detail for `tgs_select_calibration_neighborhoods`.
    Acronym tgs stands for trait gain/selection.

    Returns the negative log likelihood of held-out neutral `trait`'s gains
    under no selection, with each bin's mean branch length calibrated
    against the other neutral traits only; 0 if it has no gains or no
    calibrated exposure, so scores the same for every neighborhood size.
    """
    trait_counts_df = trait_bin_counts_df.filter(pl.col("trait") == trait)
    num_trials = trait_counts_df["num_trials"].to_numpy()
    num_successes = trait_counts_df["num_successes"].to_numpy()
    mean_bin_branch_lengths = np.divide(
        trait_counts_df["length_sum"].to_numpy(),
        num_trials,
        out=np.zeros(num_trials.shape),
        where=num_trials > 0,
    )
    # held out from its own calibration, against the other neutral traits
    assert trait in neutral_traits
    calibrated_lengths = (
        mean_bin_branch_lengths
        * tgs_calc_calibration_rescaling(
            trait_bin_counts_df=trait_bin_counts_df,
            trait=trait,
            neutral_traits=neutral_traits,
            calibration_neighborhoods=calibration_neighborhoods,
            max_rate_ratio=max_rate_ratio,
        )
    )
    exposure = (num_trials * calibrated_lengths).sum()
    if not (num_successes.any() and exposure):
        warnings.warn(
            f"neutral trait {trait_name} has no gains or no calibrated "
            "exposure, so it does not inform the choice of calibration "
            "neighborhood size.",
        )
        return 0.0
    gain_rate = num_successes.sum() / exposure  # neutral fit, to first order
    return calc_binomial_nll_proxy(
        num_trials, num_successes, gain_rate * calibrated_lengths
    )[0]
