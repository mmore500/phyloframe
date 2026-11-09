import gc
import logging
import typing

import numpy as np
import polars as pl
import polars.selectors as cs

from ..._auxlib._log_memory_usage import log_memory_usage
from ._tgs_fit_traits import tgs_fit_traits
from ._tgs_select_partitioned_calibration_neighborhoods import (
    tgs_select_partitioned_calibration_neighborhoods,
)


def _tgs_assign_calibration_bins(exposure_df: pl.DataFrame) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns each exposure episode's calibration bin and partition, one row
    per row of `exposure_df`, numbering distinct (`calibration_partition`,
    `calibration_bin`) pairs consecutively, partition-major, so that each
    partition's bins are contiguous; see Input and Output Schema.

    Input Schema
    ------------
    'calibration_partition' : integer, optional
        Each episode's calibration partition id; if missing, all 0.
    'calibration_bin' : integer, optional
        Each episode's calibration bin id; if missing, all 0. So, without
        either column, every episode is in one bin.

    Output Schema
    -------------
    'exposure_id' : integer
        Id of the episode, its row index in `exposure_df`.
    'bin' : integer
        Calibration bin index, from 0, over (partition, bin) pairs, sorted.
    'partition' : integer
        Calibration partition index, from 0, over partitions, sorted.
    """
    calibration_df = exposure_df.select(
        exposure_id=pl.int_range(pl.len(), dtype=pl.Int64),
        # defaults, overwritten by `exposure_df`'s columns, if any
        calibration_partition=pl.lit(0),
        calibration_bin=pl.lit(0),
    ).with_columns(
        exposure_df.select(
            cs.by_name(
                "calibration_partition", "calibration_bin", require_all=False
            ),
        ),
    )
    return calibration_df.select(
        "exposure_id",
        bin=pl.struct("calibration_partition", "calibration_bin").rank("dense")
        - 1,
        partition=pl.col("calibration_partition").rank("dense") - 1,
    ).cast(pl.Int64)


def tgs_estimate_tgs_from_exposures(
    exposure_df: pl.DataFrame,
    *,
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
    """Implementation detail for `alifestd_tgs_estimate_polars` and
    `alifestd_tgs_estimate_bootstrap_polars`.
    Acronym tgs stands for trait gain/selection.

    Fits each trait's gain rate and selection coefficient to the exposure
    episodes of `exposure_df` (see `_tgs_build_exposure_df`), weighted
    equally: assigns episodes to calibration bins (see
    `_tgs_assign_calibration_bins`), selects calibration neighborhoods
    within partitions (see `tgs_select_partitioned_calibration_neighborhoods`),
    and fits (see `tgs_fit_traits`). Takes validated arguments, with
    `neutral_traits` as indices and `trait_names` as labels (see
    `tgs_validate_args`).

    Returns a dict of `fits_df`, one row per trait (see Output Schema of
    `tgs_fit_traits`), and `fit_kwargs`, the `tgs_fit_traits` arguments
    but `weights`, `progress_wrap`, and `trait_bin_counts_df`, with which
    bootstrap replicates refit under other weights, holding calibration
    fixed.

    Input Schema
    ------------
    'branch_length' : float
        Branch length of each exposure table row (see
        `_tgs_build_exposure_df`).
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
    bins_df = _tgs_assign_calibration_bins(exposure_df)
    num_bins = bins_df["bin"].n_unique()
    (
        calibration_neighborhoods,
        trait_bin_counts_df,  # equally weighted, so reused by the fit
    ) = tgs_select_partitioned_calibration_neighborhoods(
        exposure_df,
        bins_df=bins_df,
        num_bins=num_bins,
        neighborhood_fractions=calibrate_neighborhood_fractions,
        neutral_traits=neutral_traits,
        trait_names=trait_names,
        max_rate_ratio=calibrate_max_rate_ratio,
    )
    distinct_survival_durations, survival_duration_indices = np.unique(
        exposure_df["survival_duration"].to_numpy(), return_inverse=True
    )
    fit_kwargs = dict(
        exposure_df=exposure_df,
        num_traits=len(trait_names),
        bins=bins_df["bin"].to_numpy(),
        num_bins=num_bins,
        calibration_neighborhoods=calibration_neighborhoods,
        distinct_survival_durations=distinct_survival_durations,
        survival_duration_indices=survival_duration_indices,
        neutral_traits=neutral_traits,
        calibrate_max_rate_ratio=calibrate_max_rate_ratio,
        length_rtol=length_rtol,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
    )
    fits_df = tgs_fit_traits(
        weights=np.ones(len(exposure_df)),
        progress_wrap=progress_wrap,
        trait_bin_counts_df=trait_bin_counts_df,
        **fit_kwargs,
    )
    del trait_bin_counts_df
    gc.collect()
    log_memory_usage(logging.debug)
    return dict(fits_df=fits_df, fit_kwargs=fit_kwargs)
