import typing
import warnings

import numpy as np
import polars as pl
from scipy import sparse

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy
from ._tgs_calc_calibration_factors import tgs_calc_calibration_factors


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
    n = trait_counts_df["num_trials"].to_numpy()
    k = trait_counts_df["num_successes"].to_numpy()
    mean_lengths = np.divide(
        trait_counts_df["length_sum"].to_numpy(),
        n,
        out=np.zeros(n.shape),
        where=n > 0,
    )
    lengths = mean_lengths * tgs_calc_calibration_factors(
        trait_bin_counts_df=trait_bin_counts_df,
        trait=trait,
        neutral_traits=neutral_traits,
        calibration_neighborhoods=calibration_neighborhoods,
        max_rate_ratio=max_rate_ratio,
    )
    exposure = (n * lengths).sum()
    if not (k.any() and exposure):
        warnings.warn(
            f"neutral trait {trait_name} has no gains or no calibrated "
            "exposure, so it does not inform the choice of calibration "
            "neighborhood size.",
        )
        return 0.0
    gain_rate = k.sum() / exposure  # neutral fit, to first order
    return calc_binomial_nll_proxy(n, k, gain_rate * lengths)[0]
