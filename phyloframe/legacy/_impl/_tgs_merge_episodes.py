import typing

import numpy as np
import polars as pl
import scipy.sparse as scipy_sparse

from ._tgs_select_trait_episodes import tgs_select_trait_episodes


def _calc_log_grid_cells(values: np.ndarray, *, rtol: float) -> np.ndarray:
    """Implementation detail for `tgs_merge_episodes`.

    Returns each nonnegative value's cell of a log-scale grid with ratio 1 +
    `rtol`; zero values get a cell of their own.
    """
    cells = np.full(len(values), np.iinfo(np.int64).min)
    is_positive = values > 0
    cells[is_positive] = np.round(np.log(values[is_positive]) / np.log1p(rtol))
    return cells


def tgs_merge_episodes(
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
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Merges `trait`'s exposure episodes (rows of `exposure_df` starting
    without the trait; see `tgs_calc_exposure_df`) whose calibrated branch
    lengths and survival durations both agree within relative tolerance
    `length_rtol`, i.e., fall in the same cell of a log-scale grid with
    ratio 1 + `length_rtol` (0 for no merging), so that merged episodes
    have nearly the same chance of gain, and merging loses little
    information about which episodes gained the trait. Each episode's
    length is calibrated by its calibration bin's factor,
    `calibration_factors[bin]`.

    Returns `partition_df`, one row per merged length cell with episodes
    (see Output Schema), and `survival_shares`, a sparse matrix with a row
    per `partition_df` row and a column per distinct survival duration
    (episodes' `survival_duration_indices`), giving the share of the row's
    episodes with each.

    Output Schema
    -------------
    'num_trials' : float
        Weighted (by `weights`) count of the row's merged episodes.
    'num_successes' : float
        Of those, the weighted count that gained the trait.
    'length' : float
        Their weighted mean calibrated branch length.
    """
    episodes_df = tgs_select_trait_episodes(
        exposure_df, trait=trait, weights=pl.Series(weights, dtype=pl.Float64)
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
    # normalize each row's episode weight over survival durations by its
    # total weight, num_trials, into shares summing to 1, so that
    # `tgs_calc_nll`'s shares @ exp(rate * distinct durations) is the row's
    # mean survival factor, exp(rate * u), not a sum growing with its
    # episode count
    survival_shares = scipy_sparse.diags(
        1 / row_sums_df["num_trials"].to_numpy()
    ) @ scipy_sparse.csr_matrix(
        (
            episode_weights,
            (
                episode_df["row"].to_numpy(),
                survival_duration_indices[episodes],
            ),
        ),
        shape=(len(row_sums_df), num_survival_durations),
    )
    partition_df = row_sums_df.select(
        "num_trials",
        "num_successes",
        pl.col("length") / pl.col("num_trials"),
    )
    return partition_df, survival_shares
