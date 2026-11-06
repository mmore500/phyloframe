import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ..._auxlib._calc_neighborhood_rate_ratios import (
    calc_neighborhood_rate_ratios,
)


def tgs_calc_calibrated_lengths(
    *,
    counts: typing.Sequence[pd.DataFrame],
    trait: int,
    neutral_traits: typing.Sequence[int],
    calibration_neighborhoods: typing.Optional[sparse.spmatrix],
    max_rate_ratio: float,
) -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns `trait`'s mean exposure episode length in each partition (zero
    where it has none), from per-trait `counts` (see `tgs_count_traits`),
    and the factors calibrating it (None without
    `calibration_neighborhoods`). Calibration rescales each partition's
    length by the rate ratio of gains of `trait`'s neutral reference, the
    other neutral traits, within the partition's neighborhood, capped at
    `max_rate_ratio`.
    """
    num_trials = counts[trait]["num_trials"].to_numpy()
    lengths = np.divide(
        counts[trait]["length_sum"].to_numpy(),
        num_trials,
        out=np.zeros(num_trials.shape),
        where=num_trials > 0,
    )
    if calibration_neighborhoods is None:
        return lengths, None

    # a neutral trait is not calibrated against itself
    references = sum(
        (counts[ref] for ref in neutral_traits if ref != trait),
        start=counts[trait] * 0,
    )
    factors = calc_neighborhood_rate_ratios(
        neighborhoods=calibration_neighborhoods,
        num_events=references["num_successes"].to_numpy(),
        exposure=references["length_sum"].to_numpy(),
        max_ratio=max_rate_ratio,
    )
    return lengths * factors, factors
