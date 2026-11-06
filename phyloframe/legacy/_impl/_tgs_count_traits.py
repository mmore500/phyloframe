import numpy as np
import polars as pl
import polars.selectors as cs

from ._tgs_extract_trait_masks import tgs_extract_trait_masks


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
    starts_absent, gains = tgs_extract_trait_masks(exposure_df)
    num_traits = starts_absent.shape[1]
    # one row per exposure episode, i.e., trait and branch starting without
    # it
    episode_traits, episodes = np.nonzero(starts_absent.T)
    episode_df = pl.DataFrame(
        {
            "trait": episode_traits,
            "bin": np.asarray(bins)[episodes],
            "weight": np.asarray(weights, dtype=np.float64)[episodes],
            "gain": gains[episodes, episode_traits],
            "length": exposure_df["branch_length"].to_numpy()[episodes],
        },
        schema_overrides={"trait": pl.Int64, "bin": pl.Int64},
    )
    # every (trait, bin) pair, including those without episodes
    all_pairs_df = pl.DataFrame(
        {"trait": np.arange(num_traits, dtype=np.int64)}
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
