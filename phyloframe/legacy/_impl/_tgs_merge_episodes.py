import typing

import numpy as np
import polars as pl
import scipy.sparse as scipy_sparse
import sklearn.preprocessing as sklearn_preprocessing

from ._tgs_extract_trait_masks import tgs_extract_trait_masks


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
    episodes, trait-major, with columns `trait` (index), the (weighted, by
    `weights`) counts of episodes, `num_trials`, and of gains,
    `num_successes`, and their mean calibrated `length`. Also returns `survival_shares`, a sparse matrix with a row
    per `partition_df` row and a column per distinct survival duration
    (episodes' `survival_duration_indices`), giving the share of the row's
    episodes with each.
    """
    starts_absent, gains = tgs_extract_trait_masks(exposure_df)
    episode_traits, episodes = np.nonzero(starts_absent.T)  # trait-major
    lengths = (
        exposure_df["branch_length"].to_numpy()[episodes]
        * calibration_factors[episode_traits, bins[episodes]]
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
    episode_weights = np.asarray(weights, dtype=np.float64)[episodes]
    episode_df = pl.DataFrame(
        {
            "trait": episode_traits,
            "length_cell": length_cells,
            "survival_cell": survival_cells,
            "weight": episode_weights,
            "gain": gains[episodes, episode_traits],
            "length": lengths,
        },
    ).with_columns(
        # each episode's merged row, numbering keys in sorted order
        row=pl.struct("trait", "length_cell", "survival_cell").rank("dense")
        - 1,
    )
    sums = (
        episode_df.group_by("row")
        .agg(
            pl.col("trait").first(),
            num_trials=pl.col("weight").sum(),
            num_successes=pl.col("weight").filter(pl.col("gain")).sum(),
            length=(pl.col("weight") * pl.col("length")).sum(),
        )
        .sort("row")
    )
    rows = episode_df["row"].to_numpy()

    # each row's distribution of episode weight over survival durations;
    # l1 normalization scales each row to sum to 1
    survival_shares = sklearn_preprocessing.normalize(
        scipy_sparse.csr_matrix(
            (episode_weights, (rows, survival_duration_indices[episodes])),
            shape=(len(sums), num_survival_durations),
        ),
        norm="l1",
    )
    # drop rows whose episodes all have zero weight (e.g., in a bootstrap
    # replicate)
    has_episodes = sums["num_trials"].to_numpy() > 0
    partition_df = sums.filter(has_episodes).select(
        "trait",
        "num_trials",
        "num_successes",
        pl.col("length") / pl.col("num_trials"),
    )
    return partition_df, survival_shares[has_episodes]
