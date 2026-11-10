import gc
import logging
import typing

import numpy as np
import polars as pl
from scipy import sparse

from ..._auxlib._calc_geometric_bins import calc_geometric_bins
from ..._auxlib._log_memory_usage import log_memory_usage
from ._tgs_calc_calibration_rescaling import (
    tgs_calc_calibration_rescaling,
)
from ._tgs_count_trait_bin_exposures import tgs_count_trait_bin_exposures
from ._tgs_fit_traits_coarse import tgs_fit_traits_coarse


def _tgs_calc_log_scale_bins(values: pl.Expr, *, rtol: float) -> pl.Expr:
    """Implementation detail for `_tgs_coarsen_episodes`.

    Returns each nonnegative value's bin on a log scale with ratio 1 +
    `rtol` (see `calc_geometric_bins`), numbering occupied bins from 0; zero
    values get bin -1.
    """

    def do_calc_bins(values: pl.Series) -> pl.Series:
        values = values.to_numpy()
        bin_ids = np.full(len(values), -1)
        is_positive = values > 0
        # number occupied bins consecutively, skipping empty ones
        __, bin_ids[is_positive] = np.unique(
            calc_geometric_bins(values[is_positive], ratio=1 + rtol),
            return_inverse=True,
        )
        return pl.Series(bin_ids)

    return values.map_batches(do_calc_bins, return_dtype=pl.Int64)


def _tgs_coarsen_episodes(
    exposure_df: pl.DataFrame,
    *,
    trait_id: int,
    calibration_factors: np.ndarray,
    length_rtol: float,
    num_survival_durations: int,
) -> typing.Tuple[pl.DataFrame, sparse.csr_matrix]:
    """Implementation detail for `tgs_fit_traits`.

    Pools `trait_id`'s exposure episodes whose calibrated lengths
    and survival durations both agree within relative tolerance
    `length_rtol` (0 for no pooling), so have nearly the same chance of
    gain. Each episode's length is calibrated by its calibration bin's
    entry of `calibration_factors`.

    Returns `coarsened_df`, one row per pool (see Output Schema), and
    `survival_duration_fractions`, a sparse matrix giving, per pool (row),
    the fraction of its episodes with each distinct survival duration
    (column, `survival_duration_id`).

    Output Schema
    -------------
    'num_trials' : float
        Weighted (by `weight`) count of the pool's episodes.
    'num_successes' : float
        Of those, the weighted count that gained the trait.
    'mean_episode_length' : float
        Their weighted mean calibrated length.
    """
    calibrated_episodes_df = exposure_df.filter(
        pl.col(f"trait{trait_id}_starts_absent"),
        pl.col("weight") > 0,  # optimization
    ).select(
        "weight",
        "survival_duration",
        "survival_duration_id",
        gained_trait=f"trait{trait_id}_gain",
        calibrated_length=pl.col("episode_length")
        * pl.lit(pl.Series(calibration_factors)).gather("bin_id"),
    )

    if length_rtol > 0:
        length_bin = _tgs_calc_log_scale_bins(
            pl.col("calibrated_length"), rtol=length_rtol
        )
        survival_bin = _tgs_calc_log_scale_bins(
            pl.col("survival_duration"), rtol=length_rtol
        )
        pool_id = pl.struct(length_bin, survival_bin).rank("dense") - 1
    else:
        pool_id = pl.int_range(pl.len())
    calibrated_episodes_df = calibrated_episodes_df.with_columns(
        pool_id=pool_id
    )

    coarsened_df = (
        calibrated_episodes_df.group_by("pool_id")
        .agg(
            num_trials=pl.col("weight").sum(),
            num_successes=pl.col("weight").filter("gained_trait").sum(),
            calibrated_length_sum=(
                pl.col("weight") * pl.col("calibrated_length")
            ).sum(),
        )
        .sort("pool_id")
        .select(
            "num_trials",
            "num_successes",
            mean_episode_length=pl.col("calibrated_length_sum")
            / pl.col("num_trials"),
        )
    )

    survival_duration_fractions = sparse.diags(
        1 / coarsened_df["num_trials"].to_numpy()
    ) @ sparse.csr_matrix(
        (
            calibrated_episodes_df["weight"].to_numpy(),
            (
                calibrated_episodes_df["pool_id"].to_numpy(),
                calibrated_episodes_df["survival_duration_id"].to_numpy(),
            ),
        ),
        shape=(len(coarsened_df), num_survival_durations),
    )
    return coarsened_df, survival_duration_fractions


def tgs_fit_traits(
    exposure_df: pl.DataFrame,
    *,
    weights: np.ndarray,
    num_traits: int,
    neutral_trait_ids: typing.Sequence[int],
    calibration_neighborhoods: sparse.spmatrix,
    calibrate_max_rate_ratio: float,
    length_rtol: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    progress_wrap: typing.Callable,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`.

    Fits each trait to the exposure episodes of `exposure_df` (see
    `tgs_build_exposure_df_polars`), each weighted by `weights` (e.g.,
    bootstrap draw counts): calibrates episode lengths by calibration bin
    (see `tgs_calc_calibration_rescaling`), pools episodes of similar
    length (see `_tgs_coarsen_episodes`), and fits gain rate and selection
    coefficient. Returns one row per trait; see `tgs_fit_traits_coarse`.
    """
    trait_bin_counts_df = tgs_count_trait_bin_exposures(
        exposure_df, trait_ids=neutral_trait_ids, weights=weights
    )

    survival_durations = exposure_df["survival_duration"].to_numpy()
    assert np.isfinite(survival_durations).all()
    distinct_survival_durations, survival_duration_ids = np.unique(
        survival_durations, return_inverse=True
    )
    exposure_df = exposure_df.with_columns(
        weight=weights, survival_duration_id=survival_duration_ids
    )

    trait_nll_kwargs = []  # each trait's, but the rate parameters
    # coarsen one trait at a time, to reduce peak memory
    for trait_id in progress_wrap(range(num_traits)):
        coarsened_df, survival_duration_fractions = _tgs_coarsen_episodes(
            exposure_df,
            trait_id=trait_id,
            calibration_factors=tgs_calc_calibration_rescaling(
                trait_bin_counts_df=trait_bin_counts_df,
                trait_id=trait_id,
                neutral_trait_ids=neutral_trait_ids,
                calibration_neighborhoods=calibration_neighborhoods,
                max_rate_ratio=calibrate_max_rate_ratio,
            ),
            length_rtol=length_rtol,
            num_survival_durations=len(distinct_survival_durations),
        )
        trait_nll_kwargs.append(
            dict(
                coarsened_df=coarsened_df,
                survival_duration_fractions=survival_duration_fractions,
                distinct_survival_durations=distinct_survival_durations,
            ),
        )

    del exposure_df, trait_bin_counts_df  # clear memory
    gc.collect()
    # called per bootstrap replicate, so log at debug level
    log_memory_usage(logging.debug)

    return tgs_fit_traits_coarse(
        trait_nll_kwargs=trait_nll_kwargs,
        neutral_trait_ids=neutral_trait_ids,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
        progress_wrap=progress_wrap,
    )
