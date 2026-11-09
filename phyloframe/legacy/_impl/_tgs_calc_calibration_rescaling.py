import typing

import numpy as np
import polars as pl
from scipy import sparse

from ..._auxlib._calc_neighborhood_rate_ratios import (
    calc_neighborhood_rate_ratios,
)


def tgs_calc_calibration_rescaling(
    *,
    trait_bin_counts_df: pl.DataFrame,
    trait_id: int,
    neutral_traits: typing.Sequence[int],
    calibration_neighborhoods: sparse.spmatrix,
    max_rate_ratio: float,
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns, for each calibration bin, the factor rescaling `trait_id`'s branch
    lengths there: the rate ratio of gains of `trait_id`'s neutral reference
    (the other neutral traits) within the bin's calibration
    neighborhood, capped at `max_rate_ratio`, given `trait_bin_counts_df` (see
    `tgs_count_trait_bin_exposures`).
    """
    # a neutral trait is not calibrated against itself
    reference_counts_df = (
        trait_bin_counts_df.filter(
            pl.col("trait_id").is_in(set(neutral_traits) - {trait_id})
        )
        .group_by("bin")
        .agg(pl.col("num_successes", "length_sum").sum())
        .sort("bin")
    )
    return calc_neighborhood_rate_ratios(
        neighborhoods=calibration_neighborhoods,
        num_events=reference_counts_df["num_successes"].to_numpy(),
        exposure=reference_counts_df["length_sum"].to_numpy(),
        max_ratio=max_rate_ratio,
    )
