import typing

import numpy as np
import polars as pl
import scipy.sparse as scipy_sparse
import sklearn.preprocessing as sklearn_preprocessing

from ._tgs_select_trait_episodes import (
    tgs_count_exposure_traits,
    tgs_select_trait_episodes,
)


def _calc_log_grid_cells(values: np.ndarray, *, rtol: float) -> np.ndarray:
    """Implementation detail for `tgs_merge_episodes`.

    Returns each nonnegative value's cell of a log-scale grid with ratio 1 +
    `rtol`; zero values get a cell of their own.
    """
    cells = np.full(len(values), np.iinfo(np.int64).min)
    is_positive = values > 0
    cells[is_positive] = np.round(np.log(values[is_positive]) / np.log1p(rtol))
    return cells


def _tgs_merge_trait_episodes(
    *,
    exposure_df: pl.DataFrame,
    trait: int,
    bins: np.ndarray,
    calibration_factors: np.ndarray,
    weights: np.ndarray,
    length_rtol: float,
    survival_duration_indices: np.ndarray,
    num_survival_durations: int,
) -> typing.Tuple[pl.DataFrame, scipy_sparse.csr_matrix]:
    """Implementation detail for `tgs_merge_episodes`.

    Returns `trait`'s rows of `partition_df` and `survival_shares` (see
    `tgs_merge_episodes`), given its `calibration_factors` per bin.
    """
    episodes_df = (
        tgs_select_trait_episodes(exposure_df, trait=trait).with_columns(
            weight=pl.lit(pl.Series(weights, dtype=pl.Float64)).gather(
                "exposure_row"
            ),
        )
        # optimization: drop zero-weight episodes
        .filter(pl.col("weight") > 0)
    )
    episodes = episodes_df["exposure_row"].to_numpy()
    lengths = (
        exposure_df["branch_length"].to_numpy()[episodes]
        * calibration_factors[bins[episodes]]
    )
    survival_durations = exposure_df["survival_duration"].to_numpy()[episodes]
    if length_rtol > 0:
        length_cells = _calc_log_grid_cells(lengths, rtol=length_rtol)
        survival_cells = _calc_log_grid_cells(
            survival_durations, rtol=length_rtol
        )
    else:
        length_cells = np.arange(len(lengths))
        survival_cells = np.zeros(len(lengths), dtype=np.int64)
    episode_weights = episodes_df["weight"].to_numpy()
    episode_df = pl.DataFrame(
        {
            "length_cell": length_cells,
            "survival_cell": survival_cells,
            "weight": episode_weights,
            "gain": episodes_df["gain"],
            "length": lengths,
        },
    ).with_columns(
        # each episode's merged row, numbering keys in sorted order
        row=pl.struct("length_cell", "survival_cell").rank("dense")
        - 1,
    )
    row_sums_df = (
        episode_df.group_by("row")
        .agg(
            num_trials=pl.col("weight").sum(),
            num_successes=pl.col("weight").filter(pl.col("gain")).sum(),
            length=(pl.col("weight") * pl.col("length")).sum(),
        )
        .sort("row")
    )
    # each row's distribution of episode weight over survival durations;
    # l1 normalization scales each row to sum to 1
    survival_shares = sklearn_preprocessing.normalize(
        scipy_sparse.csr_matrix(
            (
                episode_weights,
                (
                    episode_df["row"].to_numpy(),
                    survival_duration_indices[episodes],
                ),
            ),
            shape=(len(row_sums_df), num_survival_durations),
        ),
        norm="l1",
    )
    partition_df = row_sums_df.select(
        pl.lit(trait, dtype=pl.Int64).alias("trait"),
        "num_trials",
        "num_successes",
        pl.col("length") / pl.col("num_trials"),
    )
    return partition_df, survival_shares


def tgs_merge_episodes(
    *,
    exposure_df: pl.DataFrame,
    bins: np.ndarray,
    calibration_factors: np.ndarray,
    weights: np.ndarray,
    length_rtol: float,
    survival_duration_indices: np.ndarray,
    num_survival_durations: int,
) -> typing.Tuple[pl.DataFrame, scipy_sparse.csr_matrix]:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Merges each trait's exposure episodes (rows of `exposure_df` starting
    without the trait; see `tgs_calc_exposure_df`) whose calibrated branch
    lengths and survival durations both agree within relative tolerance
    `length_rtol`, i.e., fall in the same cell of a log-scale grid with
    ratio 1 + `length_rtol` (0 for no merging), so that merged episodes
    have nearly the same chance of gain, and merging loses little
    information about which episodes gained the trait. Each episode's
    length is calibrated by its calibration bin's factor,
    `calibration_factors[trait, bin]`.

    Returns `partition_df`, one row per trait and merged length cell with
    episodes, trait-major (see Output Schema), and `survival_shares`, a
    sparse matrix with a row per `partition_df` row and a column per
    distinct survival duration (episodes' `survival_duration_indices`),
    giving the share of the row's episodes with each.

    Output Schema
    -------------
    'trait' : integer
        Trait index.
    'num_trials' : float
        Weighted (by `weights`) count of the row's merged episodes.
    'num_successes' : float
        Of those, the weighted count that gained the trait.
    'length' : float
        Their weighted mean calibrated branch length.
    """
    trait_results = [
        # one trait at a time, bounding memory to its episodes
        _tgs_merge_trait_episodes(
            exposure_df=exposure_df,
            trait=trait,
            bins=bins,
            calibration_factors=calibration_factors[trait],
            weights=weights,
            length_rtol=length_rtol,
            survival_duration_indices=survival_duration_indices,
            num_survival_durations=num_survival_durations,
        )
        for trait in range(tgs_count_exposure_traits(exposure_df))
    ]
    return (
        pl.concat([partition_df for partition_df, __ in trait_results]),
        scipy_sparse.vstack(
            [shares for __, shares in trait_results], format="csr"
        ),
    )
