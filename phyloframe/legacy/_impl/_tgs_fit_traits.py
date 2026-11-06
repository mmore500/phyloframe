import typing

import numpy as np
import pandas as pd
from scipy import sparse

from ._tgs_calc_calibration_factors import tgs_calc_calibration_factors
from ._tgs_count_traits import tgs_count_traits
from ._tgs_fit_trait_partitions import tgs_fit_trait_partitions
from ._tgs_merge_episodes import tgs_merge_episodes


def tgs_fit_traits(
    *,
    exposure_df: pd.DataFrame,
    bins: np.ndarray,
    num_bins: int,
    calibration_neighborhoods: sparse.spmatrix,
    distinct_survival_durations: np.ndarray,
    survival_duration_indices: np.ndarray,
    weights: np.ndarray,
    neutral_traits: typing.Sequence[int],
    calibrate_max_rate_ratio: float,
    length_rtol: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
) -> pd.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Fits each trait to `exposure_df`'s exposure episodes, each weighted by
    `weights` (e.g., bootstrap draw counts): calibrates branch lengths by
    calibration bin (see `tgs_calc_calibration_factors`), merges episodes of
    similar calibrated length (see `tgs_merge_episodes`), and fits gain
    rate and selection coefficient (see `tgs_fit_trait_partitions`).
    Returns one row per trait.
    """
    counts = tgs_count_traits(
        exposure_df=exposure_df,
        bins=bins,
        num_bins=num_bins,
        weights=weights,
    )
    num_traits = len(counts.index.levels[0])
    partition_df, survival_shares = tgs_merge_episodes(
        exposure_df=exposure_df,
        bins=bins,
        calibration_factors=np.stack(
            [
                tgs_calc_calibration_factors(
                    counts=counts,
                    trait=trait,
                    neutral_traits=neutral_traits,
                    calibration_neighborhoods=calibration_neighborhoods,
                    max_rate_ratio=calibrate_max_rate_ratio,
                )
                for trait in range(num_traits)
            ],
        ),
        weights=weights,
        length_rtol=length_rtol,
        survival_duration_indices=survival_duration_indices,
        num_survival_durations=len(distinct_survival_durations),
    )
    return tgs_fit_trait_partitions(
        partition_df=partition_df,
        neutral_traits=neutral_traits,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
        survival_shares=survival_shares,
        distinct_survival_durations=distinct_survival_durations,
    )
