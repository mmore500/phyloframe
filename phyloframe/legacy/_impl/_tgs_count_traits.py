import numpy as np
import polars as pl
import polars.selectors as cs

from ._tgs_select_trait_episodes import (
    tgs_count_exposure_traits,
    tgs_select_trait_episodes,
)


def _tgs_count_trait_bins(
    *,
    exposure_df: pl.DataFrame,
    trait: int,
    bins: pl.Series,
    weights: pl.Series,
) -> pl.DataFrame:
    """Implementation detail for `tgs_count_traits`.

    Returns `trait`'s (weighted) exposure counts per calibration bin with
    episodes; see Output Schema.

    Output Schema
    -------------
    As for `tgs_count_traits`, but without 'trait', and only for bins with
    the trait's exposure episodes.
    """
    return (
        tgs_select_trait_episodes(exposure_df, trait=trait, weights=weights)
        .with_columns(
            bin=pl.lit(bins).gather("exposure_row"),
            length=pl.lit(exposure_df["branch_length"]).gather("exposure_row"),
        )
        .group_by("bin")
        .agg(
            num_trials=pl.col("weight").sum(),
            num_successes=pl.col("weight").filter(pl.col("gain")).sum(),
            length_sum=(pl.col("weight") * pl.col("length")).sum(),
        )
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
    bin_df = pl.DataFrame({"bin": np.arange(num_bins, dtype=np.int64)})
    trait_bin_kwargs = dict(  # converted once, shared by every trait
        exposure_df=exposure_df,
        bins=pl.Series(bins, dtype=pl.Int64),
        weights=pl.Series(weights, dtype=pl.Float64),
    )
    return pl.concat(
        [
            pl.DataFrame(  # in case there are no traits
                schema={
                    "trait": pl.Int64,
                    "bin": pl.Int64,
                    "num_trials": pl.Float64,
                    "num_successes": pl.Float64,
                    "length_sum": pl.Float64,
                },
            ),
            *(
                # one trait at a time, bounding memory to its episodes
                bin_df.join(
                    _tgs_count_trait_bins(trait=trait, **trait_bin_kwargs),
                    on="bin",
                    how="left",
                )
                # zero counts for bins without the trait's exposure
                # episodes, e.g., a bin whose branches all carry the trait
                .with_columns(cs.float().fill_null(0.0)).select(
                    pl.lit(trait, dtype=pl.Int64).alias("trait"), pl.all()
                )
                for trait in range(tgs_count_exposure_traits(exposure_df))
            ),
        ],
    ).sort("trait", "bin")
