import typing

import numpy as np


def tgs_normalize_gain_rates_to_neutral(
    *,
    gain_rates: np.ndarray,
    neutral_traits: typing.Sequence[int],
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns each trait's gain rate relative to the mean gain rate of its
    neutral reference, the neutral traits other than itself; nan without
    neutral reference traits, or if none of them gained their trait.
    """
    gain_rates = np.asarray(gain_rates, dtype=np.float64)

    def normalize(trait: int) -> float:
        # a neutral trait is not compared against itself
        refs = [ref for ref in neutral_traits if ref != trait]
        ref_gain_rate = gain_rates[refs].mean() if refs else 0.0
        return (
            gain_rates[trait] / ref_gain_rate if ref_gain_rate > 0 else np.nan
        )

    return np.array(
        [normalize(trait) for trait in range(len(gain_rates))],
        dtype=np.float64,
    )
