import typing

import numpy as np


def tgs_calc_neutral_reference_gain_rates(
    *,
    gain_rates: np.ndarray,
    neutral_traits: typing.Sequence[int],
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns each trait's neutral reference gain rate, the mean gain rate of
    the neutral traits other than itself, against which its gain ratio is
    taken; nan without neutral reference traits, or if none of them gained
    their trait.
    """
    gain_rates = np.asarray(gain_rates, dtype=np.float64)

    def calc_reference(trait: int) -> float:
        # a neutral trait is not compared against itself
        refs = [ref for ref in neutral_traits if ref != trait]
        ref_gain_rate = gain_rates[refs].mean() if refs else 0.0
        return ref_gain_rate if ref_gain_rate > 0 else np.nan

    return np.array(
        [calc_reference(trait) for trait in range(len(gain_rates))],
        dtype=np.float64,
    )
