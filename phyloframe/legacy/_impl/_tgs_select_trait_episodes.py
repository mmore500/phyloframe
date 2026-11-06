import polars as pl


def tgs_count_exposure_traits(exposure_df: pl.DataFrame) -> int:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns the number of traits in `exposure_df` (see
    `tgs_calc_exposure_df`).
    """
    return (
        pl.Series(exposure_df.columns, dtype=pl.String)
        .str.extract(r"^trait(\d+)_")
        .drop_nulls()
        .n_unique()
    )


def tgs_select_trait_episodes(
    exposure_df: pl.DataFrame, *, trait: int, weights: pl.Series
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns one row per exposure episode of `trait` (index) with positive
    weight, i.e., per branch of `exposure_df` (see `tgs_calc_exposure_df`)
    starting without the trait, given each branch's `weights` (e.g.,
    bootstrap draw counts); see Output Schema. Callers process one trait at
    a time, so that episode tables scale with branches, not branches times
    traits.

    Output Schema
    -------------
    'exposure_row' : integer
        Row index of the episode's branch in `exposure_df`.
    'gain' : boolean
        Whether the branch gained the trait.
    'weight' : float
        The branch's weight.
    """
    return exposure_df.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("exposure_row"),
        pl.col(f"trait{trait}_gain").alias("gain"),
        pl.lit(weights, dtype=pl.Float64).alias("weight"),
    ).filter(
        exposure_df[f"trait{trait}_starts_absent"],
        # optimization: drop zero-weight episodes
        pl.col("weight") > 0,
    )
