import typing

import numpy as np

from ._tgs_find_reference_traits import tgs_find_reference_traits


def tgs_calc_gain_ratios(
    *,
    gain_rates: np.ndarray,
    neutral_traits: typing.Sequence[int],
) -> np.ndarray:
    """Implementation detail for
    `alifestd_tgs_pendantmdc_estimate_asexual` and
    `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Returns each trait's gain rate relative to the mean gain rate of its
    neutral reference (see `tgs_find_reference_traits`); nan without
    neutral reference traits, or if none of them gained their trait.
    """
    gain_rates = np.asarray(gain_rates, dtype=np.float64)
    ratios = np.full(len(gain_rates), np.nan)
    for trait in range(len(gain_rates)):
        refs = tgs_find_reference_traits(
            neutral_traits=neutral_traits, trait=trait
        )
        ref_gain_rate = gain_rates[refs].mean() if len(refs) else 0.0
        if ref_gain_rate > 0:
            ratios[trait] = gain_rates[trait] / ref_gain_rate
    return ratios
