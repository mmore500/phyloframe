import gc
import logging
import typing

import numpy as np
import opytional as opyt
import polars as pl
import polars.selectors as cs
from scipy import sparse

from ..._auxlib._calc_geometric_bins import calc_geometric_bins
from ..._auxlib._log_memory_usage import log_memory_usage
from ._tgs_calc_calibration_rescaling import (
    tgs_calc_calibration_rescaling,
)
from ._tgs_count_trait_bin_exposures import tgs_count_trait_bin_exposures
from ._tgs_fit_coarsened_traits import tgs_fit_coarsened_traits
from ._tgs_select_trait_episodes import tgs_select_trait_episodes


def _tgs_calc_log_scale_bins(values: np.ndarray, *, rtol: float) -> np.ndarray:
    """Implementation detail for `_tgs_coarsen_episodes`.

    Returns each nonnegative value's bin id on a log scale with ratio 1 +
    `rtol` (see `calc_geometric_bins`); zero values get a bin of their own.
    """
    bin_ids = np.full(len(values), np.iinfo(np.int64).min)
    is_positive = values > 0
    bin_ids[is_positive] = calc_geometric_bins(
        values[is_positive], ratio=1 + rtol
    )
    return bin_ids


def _tgs_coarsen_episodes(
    *,
    exposure_df: pl.DataFrame,
    trait_id: int,
    bins: np.ndarray,
    calibration_factors: np.ndarray,
    weights: np.ndarray,
    length_rtol: float,
    survival_duration_indices: np.ndarray,
    num_survival_durations: int,
) -> typing.Tuple[pl.DataFrame, sparse.csr_matrix]:
    """Implementation detail for `tgs_fit_traits`. Acronym tgs stands for trait
    gain/selection.

    Coarsens `trait_id`'s exposure episodes (rows of `exposure_df` starting
    without the trait; see `_tgs_build_exposure_df`), pooling those whose
    calibrated branch lengths and survival durations both agree within
    relative tolerance `length_rtol`, i.e., fall in the same bins on a
    log scale with ratio 1 + `length_rtol` (0 for no coarsening), so
    that pooled episodes have nearly the same chance of gain, and
    coarsening loses little information about which episodes gained the
    trait. Each episode's length is calibrated by its calibration bin's
    factor, `calibration_factors[bin]`.

    Returns `coarsened_df`, one row per pair of bins with episodes (see Output
    Schema), and `survival_shares`, a sparse matrix with a row
    per `coarsened_df` row and a column per distinct survival duration
    (episodes' `survival_duration_indices`), giving the share of the row's
    episodes with each.

    Output Schema
    -------------
    'num_trials' : float
        Weighted (by `weights`) count of the row's pooled episodes.
    'num_successes' : float
        Of those, the weighted count that gained the trait.
    'mean_branch_length' : float
        Their weighted mean calibrated branch length.
    """
    episodes_df = tgs_select_trait_episodes(
        exposure_df,
        trait_id=trait_id,
        weights=pl.Series(weights, dtype=pl.Float64),
    )
    episodes = episodes_df["exposure_id"].to_numpy()
    lengths = (
        exposure_df["branch_length"].to_numpy()[episodes]
        * calibration_factors[bins[episodes]]
    )
    survival_durations = exposure_df["survival_duration"].to_numpy()[episodes]
    if length_rtol > 0:
        length_bin_ids = _tgs_calc_log_scale_bins(lengths, rtol=length_rtol)
        survival_bin_ids = _tgs_calc_log_scale_bins(
            survival_durations, rtol=length_rtol
        )
    else:
        length_bin_ids = np.arange(len(lengths))
        survival_bin_ids = np.zeros(len(lengths), dtype=np.int64)
    episode_weights = episodes_df["weight"].to_numpy()
    episode_df = pl.DataFrame(
        {
            "length_bin_id": length_bin_ids,
            "survival_bin_id": survival_bin_ids,
            "weight": episode_weights,
            "gain": episodes_df["gain"],
            "length": lengths,
        },
    ).with_columns(
        # each episode's coarsened row, numbering keys in sorted order
        row=pl.struct("length_bin_id", "survival_bin_id").rank(method="dense")
        - 1,
    )
    # copied into episode_df; free before aggregating
    del episodes_df, lengths, survival_durations
    del length_bin_ids, survival_bin_ids
    row_sums_df = (
        episode_df.group_by("row")
        .agg(
            num_trials=pl.col("weight").sum(),
            num_successes=pl.col("weight").filter(pl.col("gain")).sum(),
            length=(pl.col("weight") * pl.col("length")).sum(),
        )
        .sort("row")
    )
    survival_shares = sparse.diags(
        1 / row_sums_df["num_trials"].to_numpy()
    ) @ sparse.csr_matrix(
        (
            episode_weights,
            (
                episode_df["row"].to_numpy(),
                survival_duration_indices[episodes],
            ),
        ),
        shape=(len(row_sums_df), num_survival_durations),
    )
    assert np.allclose(survival_shares.sum(axis=1), 1.0)
    coarsened_df = row_sums_df.select(
        "num_trials",
        "num_successes",
        mean_branch_length=pl.col("length") / pl.col("num_trials"),
    )
    return coarsened_df, survival_shares


def tgs_fit_traits(
    *,
    exposure_df: pl.DataFrame,
    num_traits: int,
    bins: np.ndarray,
    num_bins: int,
    calibration_neighborhoods: sparse.spmatrix,
    distinct_survival_durations: np.ndarray,
    survival_duration_indices: np.ndarray,
    weights: np.ndarray,
    neutral_trait_ids: typing.Sequence[int],
    calibrate_max_rate_ratio: float,
    length_rtol: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    growth_exponent_bound: float,
    progress_wrap: typing.Callable,
    trait_bin_counts_df: typing.Optional[pl.DataFrame],
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Fits each trait to `exposure_df`'s exposure episodes, each weighted by
    `weights` (e.g., bootstrap draw counts): calibrates branch lengths by
    calibration bin (see `tgs_calc_calibration_rescaling`), pools episodes
    of similar calibrated length (see `_tgs_coarsen_episodes`), and fits gain
    rate and selection coefficient (see `tgs_fit_coarsened_traits`).

    Coarsens one trait at a time, each wrapped in `progress_wrap` (e.g.,
    tqdm), so that episode tables scale with branches, not branches times
    traits; every trait's coarsened episodes, far fewer with coarsening
    (`length_rtol` > 0), are kept for fitting. Pass `trait_bin_counts_df`
    (see `tgs_count_trait_bin_exposures`) if already computed for these
    `weights`.

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

    'trait_id', 'bin' : integer
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
    assert (
        num_traits == exposure_df.select(cs.matches(r"^trait\d+_gain$")).width
    )
    trait_bin_counts_df = opyt.or_else(
        trait_bin_counts_df,
        lambda: tgs_count_trait_bin_exposures(
            exposure_df=exposure_df,
            num_traits=num_traits,
            bins=bins,
            num_bins=num_bins,
            weights=weights,
        ),
    )
    trait_nll_kwargs = []  # each trait's, but the rate parameters
    for trait_id in progress_wrap(range(num_traits)):
        coarsened_df, survival_shares = _tgs_coarsen_episodes(
            exposure_df=exposure_df,
            trait_id=trait_id,
            bins=bins,
            calibration_factors=tgs_calc_calibration_rescaling(
                trait_bin_counts_df=trait_bin_counts_df,
                trait_id=trait_id,
                neutral_trait_ids=neutral_trait_ids,
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
                coarsened_df=coarsened_df,
                survival_shares=survival_shares,
                distinct_survival_durations=distinct_survival_durations,
            ),
        )
    del trait_bin_counts_df
    gc.collect()

    # called per bootstrap replicate, so log at debug level
    log_memory_usage(logging.debug)
    return tgs_fit_coarsened_traits(
        trait_nll_kwargs=trait_nll_kwargs,
        neutral_trait_ids=neutral_trait_ids,
        loss_rate=loss_rate,
        selection_coef_starts=selection_coef_starts,
        growth_exponent_bound=growth_exponent_bound,
        progress_wrap=progress_wrap,
    )
