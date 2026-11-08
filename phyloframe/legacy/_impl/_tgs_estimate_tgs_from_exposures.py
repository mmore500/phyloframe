import gc
import logging
import typing

import numpy as np
import polars as pl
from scipy import sparse

from ..._auxlib._log_memory_usage import log_memory_usage
from ._tgs_assign_calibration_bins import tgs_assign_calibration_bins
from ._tgs_fit_traits import tgs_fit_traits
from ._tgs_select_partitioned_calibration_neighborhoods import (
    tgs_select_partitioned_calibration_neighborhoods,
)


def tgs_estimate_tgs_from_exposures(
    exposure_df: pl.DataFrame,
    *,
    calibration_neighborhoods: typing.Optional[sparse.spmatrix],
    weights: typing.Optional[np.ndarray],
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_max_rate_ratio: float,
    length_rtol: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    progress_wrap: typing.Callable,
) -> typing.Dict[str, typing.Any]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`, and their bootstrap variants.
    Acronym tgs stands for trait gain/selection.

    Fits each trait's gain rate and selection coefficient once to the
    exposure episodes of `exposure_df` (see `tgs_build_exposure_df`):
    assigns episodes to calibration bins (see `tgs_assign_calibration_bins`),
    selects calibration neighborhoods within partitions (see
    `tgs_select_partitioned_calibration_neighborhoods`), and fits (see
    `tgs_fit_traits`). Takes validated arguments, with `neutral_traits` as
    indices and `trait_names` as labels (see `tgs_validate_args`).

    For a point estimate, pass `calibration_neighborhoods` and `weights` as
    None: neighborhoods are selected, and episodes weighted equally. For a
    bootstrap replicate, pass both: the point estimate's neighborhoods, so
    calibration is held fixed, and the replicate's episode weights (e.g.,
    bootstrap draw counts).

    Returns a dict of `fits_df`, one row per trait (see Output Schema of
    `tgs_fit_traits`), and `calibration_neighborhoods`, those used.

    Input Schema
    ------------
    'branch_length' : float
        Branch length of each exposure table row (see
        `tgs_build_exposure_df`).
    'survival_duration' : float
        Time from the branch's end node to the latest leaf below it.
    'trait{i}_starts_absent', 'trait{i}_gain' : boolean, per trait `i`
        Whether the branch started without, and whether it gained, the
        trait.
    'calibration_partition', 'calibration_bin' : integer, optional
        The branch's calibration partition and bin.
    'calibration_compare_by_{name}' : numeric, optional
        Values by which calibration bins are compared.
    """
    # neighborhoods are selected from equally weighted episodes, so a
    # weighted fit must reuse a point estimate's
    assert len({x is None for x in (calibration_neighborhoods, weights)}) == 1

    bins_df = tgs_assign_calibration_bins(exposure_df)
    bins = bins_df["bin"].to_numpy()
    num_episodes, num_bins = len(exposure_df), bins_df["bin"].n_unique()
    trait_bin_counts_df = None  # counted with `weights` by tgs_fit_traits
    if calibration_neighborhoods is None:
        (
            calibration_neighborhoods,
            trait_bin_counts_df,  # equally weighted, so reusable
        ) = tgs_select_partitioned_calibration_neighborhoods(
            exposure_df,
            bins_df=bins_df,
            num_bins=num_bins,
            neighborhood_fractions=calibrate_neighborhood_fractions,
            neutral_traits=neutral_traits,
            trait_names=trait_names,
            max_rate_ratio=calibrate_max_rate_ratio,
        )
        weights = np.ones(num_episodes)

    distinct_survival_durations, survival_duration_indices = np.unique(
        exposure_df["survival_duration"].to_numpy(), return_inverse=True
    )
    fits_df = tgs_fit_traits(
        exposure_df=exposure_df,
        bins=bins,
        num_bins=num_bins,
        calibration_neighborhoods=calibration_neighborhoods,
        distinct_survival_durations=distinct_survival_durations,
        survival_duration_indices=survival_duration_indices,
        weights=weights,
        neutral_traits=neutral_traits,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
        progress_wrap=progress_wrap,
        trait_bin_counts_df=trait_bin_counts_df,
    )
    del trait_bin_counts_df
    gc.collect()
    log_memory_usage(logging.debug)
    return dict(
        fits_df=fits_df, calibration_neighborhoods=calibration_neighborhoods
    )
