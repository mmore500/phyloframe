import polars as pl


def tgs_select_trait_episodes(
    exposure_df: pl.DataFrame, *, trait_id: int, weights: pl.Series
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns one row per exposure episode of `trait_id` with positive
    weight, i.e., per branch of `exposure_df` (see `_tgs_build_exposure_df`)
    starting without the trait, given each branch's `weights` (e.g.,
    bootstrap draw counts); see Output Schema. Callers process one trait at
    a time, so that episode tables scale with branches, not branches times
    traits.

    Output Schema
    -------------
    'exposure_id' : integer
        Id of the episode, its row index in `exposure_df`.
    'gain' : boolean
        Whether the branch gained the trait.
    'weight' : float
        The branch's weight.
    """
    return exposure_df.select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("exposure_id"),
        pl.col(f"trait{trait_id}_gain").alias("gain"),
        pl.lit(weights, dtype=pl.Float64).alias("weight"),
    ).filter(
        exposure_df[f"trait{trait_id}_starts_absent"],
        # optimization: drop zero-weight episodes
        pl.col("weight") > 0,
    )
