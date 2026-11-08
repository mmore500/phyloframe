import typing

import numpy as np


def _tgs_calc_neutral_reference_gain_rate(
    *,
    trait_id: int,
    gain_rates: np.ndarray,
    neutral_traits: typing.Sequence[int],
) -> float:
    """Implementation detail for `tgs_calc_neutral_reference_gain_rates`.
    Acronym tgs stands for trait gain/selection.

    Returns trait `trait_id`'s neutral reference gain rate, the mean gain
    rate of the neutral traits other than itself; nan if none of them
    gained their trait.
    """
    # a neutral trait is not compared against itself
    refs = [ref for ref in neutral_traits if ref != trait_id]
    ref_gain_rate = gain_rates[refs].mean()
    return ref_gain_rate if ref_gain_rate > 0 else np.nan


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
    taken; nan if none of them gained their trait.

    Raises ValueError with fewer than two distinct neutral traits, so that
    every trait has a neutral reference.
    """
    if len(set(neutral_traits)) < 2:
        raise ValueError(
            "at least two distinct neutral traits are required, so that "
            "every trait has a neutral reference",
        )
    gain_rates = np.asarray(gain_rates, dtype=np.float64)
    return np.array(
        [
            _tgs_calc_neutral_reference_gain_rate(
                trait_id=trait_id,
                gain_rates=gain_rates,
                neutral_traits=neutral_traits,
            )
            for trait_id in range(len(gain_rates))
        ],
        dtype=np.float64,
    )
