import numpy as np
import polars as pl
import polars.selectors as cs

from ._tgs_unpivot_exposure_episodes import (
    tgs_count_exposure_traits,
    tgs_unpivot_exposure_episodes,
)


def tgs_count_traits(
    *,
    exposure_df: pl.DataFrame,
    bins: np.ndarray,
    num_bins: int,
    weights: np.ndarray,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns each trait's exposure counts in each calibration bin, one row
    per trait and bin, trait-major, given each row of `exposure_df` (see
    `tgs_calc_exposure_df`), its `bins` index, and its `weights` (e.g.,
    bootstrap draw counts); see Output Schema.

    Output Schema
    -------------
    'trait' : integer
        Trait index.
    'bin' : integer
        Calibration bin index, from 0 to `num_bins` - 1.
    'num_trials' : float
        Weighted count of the trait's exposure episodes in the bin.
    'num_successes' : float
        Of those, the weighted count that gained the trait.
    'length_sum' : float
        Weighted sum of those episodes' branch lengths.
    """
    weights = np.asarray(weights, dtype=np.float64)
    episode_df = tgs_unpivot_exposure_episodes(exposure_df)
    # optimization: drop zero-weight episodes
    episode_df = episode_df.filter(
        weights[episode_df["exposure_row"].to_numpy()] > 0
    )
    exposure_rows = episode_df["exposure_row"].to_numpy()
    episode_df = episode_df.with_columns(
        bin=np.asarray(bins, dtype=np.int64)[exposure_rows],
        weight=weights[exposure_rows],
        length=exposure_df["branch_length"].to_numpy()[exposure_rows],
    )
    # every (trait, bin) pair, including those without episodes
    all_pairs_df = pl.DataFrame(
        {
            "trait": np.arange(
                tgs_count_exposure_traits(exposure_df), dtype=np.int64
            )
        }
    ).join(
        pl.DataFrame({"bin": np.arange(num_bins, dtype=np.int64)}),
        how="cross",
    )
    return (
        all_pairs_df.join(
            episode_df.group_by("trait", "bin").agg(
                num_trials=pl.col("weight").sum(),
                num_successes=pl.col("weight").filter(pl.col("gain")).sum(),
                length_sum=(pl.col("weight") * pl.col("length")).sum(),
            ),
            on=["trait", "bin"],
            how="left",
        )
        # zero counts for (trait, bin) pairs without exposure episodes, e.g.,
        # a bin whose exposure_rows all carry the trait
        .with_columns(cs.float().fill_null(0.0)).sort("trait", "bin")
    )
