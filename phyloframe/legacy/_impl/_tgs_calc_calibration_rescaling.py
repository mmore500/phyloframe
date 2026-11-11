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
    neutral_trait_ids: typing.Sequence[int],
    calibration_neighborhoods: sparse.spmatrix,
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_estimate_polars`.

    Returns, for each calibration bin, the factor rescaling `trait_id`'s
    episode lengths there: the gain rate of the neutral traits other than
    `trait_id` within the bin's calibration neighborhood, relative to
    overall (see `calc_neighborhood_rate_ratios`). Takes the neutral traits'
    `trait_bin_counts_df` (see `tgs_count_trait_bin_exposures`).
    """
    reference_counts_df = (
        # exclude neutral traits from own calibration set
        trait_bin_counts_df.filter(
            pl.col("trait_id").is_in(set(neutral_trait_ids) - {trait_id})
        )
        .group_by("bin_id")
        .agg(pl.col("num_gains", "total_episode_duration").sum())
        .sort("bin_id")
    )
    return calc_neighborhood_rate_ratios(
        neighborhoods=calibration_neighborhoods,
        num_events=reference_counts_df["num_gains"].to_numpy(),
        exposure=reference_counts_df["total_episode_duration"].to_numpy(),
        pseudocount=1.0,  # stabilizes calibration against sparse gains
    )
