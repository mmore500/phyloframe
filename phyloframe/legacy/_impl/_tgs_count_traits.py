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

    Returns, with columns `trait` and `bin`, trait-major, each trait's
    (weighted) count of exposure episodes in each calibration bin,
    `num_trials`, of their gains, `num_successes`, and their summed branch
    length, `length_sum`, given each row of `exposure_df` (see
    `tgs_calc_exposure_df`), its `bins` index, and its `weights` (e.g.,
    bootstrap draw counts).
    """
    episode_df = tgs_unpivot_exposure_episodes(exposure_df)
    branches = episode_df["branch"].to_numpy()
    episode_df = episode_df.with_columns(
        bin=np.asarray(bins, dtype=np.int64)[branches],
        weight=np.asarray(weights, dtype=np.float64)[branches],
        length=exposure_df["branch_length"].to_numpy()[branches],
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
        .with_columns(cs.float().fill_null(0.0))
        .sort("trait", "bin")
    )
