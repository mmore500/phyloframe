import typing
import warnings

from scipy import sparse

from ..._auxlib._calc_binomial_nll_proxy import calc_binomial_nll_proxy
from ._TgsCounts import TgsCounts
from ._tgs_calc_calibrated_pendant_lengths import (
    tgs_calc_calibrated_pendant_lengths,
)


def tgs_calc_heldout_nll(
    *,
    calibration_neighborhoods: sparse.spmatrix,
    counts: TgsCounts,
    neutral_traits: typing.Sequence[int],
    trait_names: typing.Sequence[str],
) -> float:
    """Implementation detail for `tgs_select_calibration_neighborhoods`.

    Here, tgs stands for trait gain/selection.

    Returns summed negative log likelihood of each neutral trait's carriers
    under no selection, with pendant lengths calibrated against the other
    neutral traits only.
    """
    res = 0.0
    for trait in neutral_traits:
        n = counts.num_leaves[trait]
        k = counts.num_carriers[trait]
        pendant_length, __ = tgs_calc_calibrated_pendant_lengths(
            counts=counts,
            trait=trait,
            neutral_traits=neutral_traits,
            calibration_neighborhoods=calibration_neighborhoods,
        )
        exposure = (n * pendant_length).sum()
        if k.any() and exposure:
            gain_rate = k.sum() / exposure  # neutral fit, to first order
            res += calc_binomial_nll_proxy(n, k, gain_rate * pendant_length)[0]
        else:  # scores the same for every neighborhood size
            warnings.warn(
                f"neutral trait {trait_names[trait]} has no carriers or no "
                "calibrated exposure, so it does not inform the choice of "
                "calibration neighborhood size.",
            )
    return res
