import typing
import warnings

import pandas as pd

from scipy import sparse

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy
from ._tgs_calc_calibrated_lengths import (
    tgs_calc_calibrated_lengths,
)


def tgs_calc_heldout_nll(
    *,
    calibration_neighborhoods: sparse.spmatrix,
    counts: typing.Sequence[pd.DataFrame],
    trait: int,
    trait_name: str,
    neutral_traits: typing.Sequence[int],
    max_rate_ratio: float,
) -> float:
    """Implementation detail for `tgs_select_calibration_neighborhoods`.
    Acronym tgs stands for trait gain/selection.

    Returns the negative log likelihood of held-out neutral `trait`'s
    carriers under no selection, with pendant lengths calibrated against
    the other neutral traits only; 0 if it has no carriers or no calibrated
    exposure, so scores the same for every neighborhood size.
    """
    n = counts[trait]["num_trials"].to_numpy()
    k = counts[trait]["num_successes"].to_numpy()
    pendant_length, __ = tgs_calc_calibrated_lengths(
        counts=counts,
        trait=trait,
        neutral_traits=neutral_traits,
        calibration_neighborhoods=calibration_neighborhoods,
        max_rate_ratio=max_rate_ratio,
    )
    exposure = (n * pendant_length).sum()
    if not (k.any() and exposure):
        warnings.warn(
            f"neutral trait {trait_name} has no carriers or no calibrated "
            "exposure, so it does not inform the choice of calibration "
            "neighborhood size.",
        )
        return 0.0
    gain_rate = k.sum() / exposure  # neutral fit, to first order
    return calc_binomial_nll_proxy(n, k, gain_rate * pendant_length)[0]
