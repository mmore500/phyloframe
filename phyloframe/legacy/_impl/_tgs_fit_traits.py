import typing

import numpy as np
import polars as pl
from scipy import sparse

from ..._auxlib._identity import identity
from ._tgs_calc_calibration_factors import tgs_calc_calibration_factors
from ._tgs_count_traits import tgs_count_traits
from ._tgs_fit_trait_partitions import tgs_fit_trait_partitions
from ._tgs_merge_episodes import tgs_merge_episodes
from ._tgs_select_trait_episodes import tgs_count_exposure_traits


def tgs_fit_traits(
    *,
    exposure_df: pl.DataFrame,
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
    progress_wrap: typing.Callable = identity,
    trait_bin_counts_df: typing.Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Fits each trait to `exposure_df`'s exposure episodes, each weighted by
    `weights` (e.g., bootstrap draw counts): calibrates branch lengths by
    calibration bin (see `tgs_calc_calibration_factors`), merges episodes of
    similar calibrated length (see `tgs_merge_episodes`), and fits gain
    rate and selection coefficient (see `tgs_fit_trait_partitions`).
    Merges, then fits, one trait at a time, each wrapped in `progress_wrap`
    (e.g., tqdm), so that episode tables scale with branches, not branches
    times traits; every trait's merged partitions, far fewer with merging
    (`length_rtol` > 0), are kept for fitting. Pass `trait_bin_counts_df`
    (see `tgs_count_traits`) if already computed for these `weights`.
    Returns one row per trait; see Input and Output Schema.

    Input Schema
    ------------
    `exposure_df` columns:

    'branch_length' : float
        Branch length.
    'survival_duration' : float
        Time from the branch's end node to the latest leaf below it.
    'trait{i}_starts_absent', 'trait{i}_gain' : boolean, per trait `i`
        Whether the branch started without, and whether it gained, the
        trait.

    `trait_bin_counts_df` columns, optional:

    'trait', 'bin' : integer
        Trait and calibration bin indices, one row per pair, trait-major.
    'num_trials', 'num_successes', 'length_sum' : float
        (Weighted) count of the trait's exposure episodes in the bin, of
        those that gained it, and their summed branch length.

    Output Schema
    -------------
    'gain_rate' : float
        Fitted gain rate, per unit branch length; 0 without gains.
    'selection_coef' : float
        Fitted selection coefficient; nan without gains.
    'selection_coef_lrt_naive' : float
        Likelihood-ratio statistic for selection coefficient = 0.
    'gain_ratio' : float
        Gain rate relative to the mean gain rate of the trait's neutral
        reference.
    'gain_ratio_lrt_naive' : float
        Likelihood-ratio statistic for gain ratio = 1.
    """
    if trait_bin_counts_df is None:
        trait_bin_counts_df = tgs_count_traits(
            exposure_df=exposure_df,
            bins=bins,
            num_bins=num_bins,
            weights=weights,
        )
    trait_nll_kwargs = []  # each trait's, but the rate parameters
    for trait in progress_wrap(range(tgs_count_exposure_traits(exposure_df))):
        partitions, survival_shares = tgs_merge_episodes(
            exposure_df=exposure_df,
            trait=trait,
            bins=bins,
            calibration_factors=tgs_calc_calibration_factors(
                trait_bin_counts_df=trait_bin_counts_df,
                trait=trait,
                neutral_traits=neutral_traits,
                calibration_neighborhoods=calibration_neighborhoods,
                max_rate_ratio=calibrate_max_rate_ratio,
            ),
            weights=weights,
            length_rtol=length_rtol,
            survival_duration_indices=survival_duration_indices,
            num_survival_durations=len(distinct_survival_durations),
        )
        trait_nll_kwargs.append(
            dict(
                partitions=partitions,
                growth_exponent_bound=growth_exponent_bound,
                survival_shares=survival_shares,
                distinct_survival_durations=distinct_survival_durations,
            ),
        )
    return tgs_fit_trait_partitions(
        trait_nll_kwargs=trait_nll_kwargs,
        neutral_traits=neutral_traits,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        progress_wrap=progress_wrap,
    )
