import functools
import logging
import typing

import numpy as np
import polars as pl
from scipy import sparse

from ..._auxlib._calc_group_means import calc_group_means
from ..._auxlib._calc_tiebroken_ranks import calc_tiebroken_ranks
from ._tgs_bootstrap_trait_fits import tgs_bootstrap_trait_fits
from ._tgs_build_estimates_df import tgs_build_estimates_df
from ._tgs_count_traits import tgs_count_traits
from ._tgs_fit_traits import tgs_fit_traits
from ._tgs_select_calibration_neighborhoods import (
    tgs_select_calibration_neighborhoods,
)


def _slice_bins(counts: pl.DataFrame, begin: int, end: int) -> pl.DataFrame:
    """Implementation detail for `tgs_impl`.

    Returns `counts` (see `tgs_count_traits`) of bins `begin` up to `end`,
    renumbered from 0.
    """
    return counts.filter(
        pl.col("bin").is_between(begin, end - 1)
    ).with_columns(pl.col("bin") - begin)


def tgs_impl(
    exposure_df: pl.DataFrame,
    *,
    counts_df: pl.DataFrame,
    trait_names: typing.Sequence[str],
    neutral_traits: typing.Sequence[int],
    calibrate_neighborhood_fractions: typing.Sequence[float],
    calibrate_max_rate_ratio: float,
    length_rtol: float,
    num_bootstrap: int,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    seed: typing.Optional[int],
    parallel_backend: typing.Optional[str],
    progress_wrap: typing.Callable,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Fits each trait to the exposure episodes of `exposure_df` (see
    `tgs_calc_exposure_df`), calibrating by its `calibration_bin` column,
    if any, comparing bins by their mean ranks in its
    `calibration_compare_by_*` columns, only within the same
    `calibration_partition`, and returns per-trait estimates following
    `counts_df`'s per-trait counts (see `tgs_build_estimates_df`). Takes
    validated arguments, with `neutral_traits` as indices and `trait_names`
    as labels (see `tgs_validate_args`).
    """
    num_episodes = len(exposure_df)
    if not num_episodes:
        raise ValueError("phylogeny_df has no exposure episodes to fit")

    # number (partition, bin) pairs consecutively, partition-major, so that
    # each partition's bins are contiguous; without columns, all are one
    zeros = np.zeros(num_episodes, dtype=np.int64)
    pairs, bins = np.unique(
        np.column_stack(
            [
                (
                    exposure_df[column].to_numpy()
                    if column in exposure_df.columns
                    else zeros
                )
                for column in ("calibration_partition", "calibration_bin")
            ],
        ),
        axis=0,
        return_inverse=True,
    )
    bins = bins.ravel()
    num_bins = len(pairs)
    if num_bins > 1:
        coordinates = [
            exposure_df[column].to_numpy()
            for column in exposure_df.columns
            if column.startswith("calibration_compare_by_")
        ]
        # bins' mean ranks; constant coordinates carry no distance
        ranks = calc_tiebroken_ranks(
            [x for x in coordinates if np.ptp(x) > 0] or coordinates[:1]
        )
        bin_coords = calc_group_means(
            bins, ranks / num_episodes, num_groups=num_bins
        )
        counts = tgs_count_traits(
            exposure_df=exposure_df,
            bins=bins,
            num_bins=num_bins,
            weights=np.ones(num_episodes),
        )
        # neighborhoods never span partitions, selected separately within
        # each
        __, partition_starts = np.unique(pairs[:, 0], return_index=True)
        partition_bounds = [*partition_starts, num_bins]
        calibration_neighborhoods = sparse.block_diag(
            [
                tgs_select_calibration_neighborhoods(
                    bin_coords=bin_coords[begin:end],
                    neighborhood_fractions=calibrate_neighborhood_fractions,
                    counts=_slice_bins(counts, begin, end),
                    neutral_traits=neutral_traits,
                    trait_names=trait_names,
                    max_rate_ratio=calibrate_max_rate_ratio,
                )
                for begin, end in zip(partition_bounds, partition_bounds[1:])
            ],
            format="csr",
        )
    else:  # one neighborhood of all episodes, i.e., no calibration
        calibration_neighborhoods = sparse.identity(1, format="csr")

    distinct_survival_durations, survival_duration_indices = np.unique(
        exposure_df["survival_duration"].to_numpy(), return_inverse=True
    )
    fit_kwargs = dict(
        exposure_df=exposure_df,
        bins=bins,
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
    logging.info(
        f"- tgs: fitting {len(trait_names)} traits to {num_episodes} "
        "exposure episodes...",
    )
    point_estimate_df = tgs_fit_traits(
        weights=np.ones(num_episodes), **fit_kwargs
    )
    bootstrap_fits_df = None
    if num_bootstrap:
        logging.info(f"- tgs: fitting {num_bootstrap} bootstrap replicates...")
        bootstrap_fits_df = tgs_bootstrap_trait_fits(
            do_fit=functools.partial(tgs_fit_traits, **fit_kwargs),
            num_items=num_episodes,
            num_bootstrap=num_bootstrap,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )
    return tgs_build_estimates_df(
        counts_df,
        point_estimate_df=point_estimate_df,
        bootstrap_fits_df=bootstrap_fits_df,
        num_bootstrap=num_bootstrap,
    )
