import typing

import numpy as np
import polars as pl
from scipy import sparse

from ..._auxlib._calc_neighborhood_rate_ratios import (
    calc_neighborhood_rate_ratios,
)


def tgs_calc_calibration_factors(
    *,
    counts: pl.DataFrame,
    trait: int,
    neutral_traits: typing.Sequence[int],
    calibration_neighborhoods: sparse.spmatrix,
    max_rate_ratio: float,
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns, for each calibration bin, the factor rescaling `trait`'s branch
    lengths there: the rate ratio of gains of `trait`'s neutral reference
    (the other neutral traits) within the bin's calibration
    neighborhood, capped at `max_rate_ratio`, given `counts` (see
    `tgs_count_traits`).
    """
    # a neutral trait is not calibrated against itself
    references = (
        counts.filter(
            pl.col("trait").is_in(
                [ref for ref in neutral_traits if ref != trait]
            )
        )
        .group_by("bin")
        .agg(pl.col("num_successes", "length_sum").sum())
        .sort("bin")
    )
    return calc_neighborhood_rate_ratios(
        neighborhoods=calibration_neighborhoods,
        num_events=references["num_successes"].to_numpy(),
        exposure=references["length_sum"].to_numpy(),
        max_ratio=max_rate_ratio,
    )
