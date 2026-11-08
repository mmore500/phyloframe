import functools
import gc
import logging
import typing

import numpy as np
import polars as pl

from ..._auxlib._log_memory_usage import log_memory_usage
from ._tgs_assign_calibration_bins import tgs_assign_calibration_bins
from ._tgs_bootstrap_trait_fits import tgs_bootstrap_trait_fits
from ._tgs_build_estimates_df import tgs_build_estimates_df
from ._tgs_fit_traits import tgs_fit_traits
from ._tgs_select_partitioned_calibration_neighborhoods import (
    tgs_select_partitioned_calibration_neighborhoods,
)


def tgs_estimate_tgs_from_exposures(
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

    Estimates each trait's gain rate and selection coefficient from the
    exposure episodes of `exposure_df` (see `tgs_build_exposure_df`):
    assigns episodes to calibration bins (see `tgs_assign_calibration_bins`),
    selects calibration neighborhoods within partitions (see
    `tgs_select_partitioned_calibration_neighborhoods`), fits point
    estimates and any bootstrap replicates (see `tgs_fit_traits`), and
    returns per-trait estimates following `counts_df`'s per-trait counts
    (see `tgs_build_estimates_df`). Takes
    validated arguments, with `neutral_traits` as indices and `trait_names`
    as labels (see `tgs_validate_args`); see Input and Output Schema.

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
    `counts_df` columns
        Trait name, 'trait', then per-trait counts, e.g., as for
        `tgs_tally_exposure_episodes`.

    Output Schema
    -------------
    `counts_df` columns
        Trait name, 'trait', then per-trait counts.
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
    'selection_coef_se', 'log_gain_ratio_se' : float, with `num_bootstrap`
        Bootstrap standard errors of the selection coefficient and log gain
        ratio.
    'selection_coef_z', 'gain_ratio_z' : float, with `num_bootstrap`
        Selection coefficient and log gain ratio over their standard
        errors.
    'selection_coef_p', 'gain_ratio_p' : float, with `num_bootstrap`
        Two-sided normal p-values of those z-scores.
    'num_bootstrap_degenerate' : integer, with `num_bootstrap`
        Replicates excluded for an undefined selection coefficient or gain
        ratio.
    """
    bins_df = tgs_assign_calibration_bins(exposure_df)
    bins = bins_df["bin"].to_numpy()
    num_episodes, num_bins = len(exposure_df), int(bins.max()) + 1
    (
        calibration_neighborhoods,
        trait_bin_counts_df,
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
    # progress over traits for the point estimate; over replicates, not
    # each replicate's traits, for the bootstrap
    point_estimate_df = tgs_fit_traits(
        weights=np.ones(num_episodes),
        progress_wrap=progress_wrap,
        trait_bin_counts_df=trait_bin_counts_df,
        **fit_kwargs,
    )
    del trait_bin_counts_df  # replicates count with their own weights
    gc.collect()
    log_memory_usage(logging.info)
    bootstrap_fits_df = None
    if num_bootstrap:
        logging.info(
            f"- tgs: fitting {num_bootstrap} bootstrap replicates...",
        )
        bootstrap_fits_df = tgs_bootstrap_trait_fits(
            do_fit=functools.partial(tgs_fit_traits, **fit_kwargs),
            num_items=num_episodes,
            num_bootstrap=num_bootstrap,
            seed=seed,
            parallel_backend=parallel_backend,
            progress_wrap=progress_wrap,
        )
        gc.collect()
        log_memory_usage(logging.info)
    return tgs_build_estimates_df(
        counts_df,
        point_estimate_df=point_estimate_df,
        bootstrap_fits_df=bootstrap_fits_df,
    )
