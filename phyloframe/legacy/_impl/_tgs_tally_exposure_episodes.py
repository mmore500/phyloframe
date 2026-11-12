import typing

import polars as pl


def tgs_tally_exposure_episodes(
    exposure_df: pl.DataFrame, *, trait_names: typing.Sequence[str]
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`.

    Returns one row per trait, with counts of its exposure episodes in
    `exposure_df` (see `tgs_build_exposure_df_polars`).

    Output Schema
    -------------
    'num_leaf_episodes' : integer
        Leaf branches that started without the trait.
    'num_leaf_gains' : integer
        Of those, the number that gained the trait.
    'num_inner_episodes' : integer
        Inner branches that started without the trait.
    'num_inner_gains' : integer
        Of those, the number that gained the trait.
    'trait_name' : string
        Trait name.
    """
    is_leaf = pl.col("clade_size") == 1

    def tally_trait(trait_id: int) -> pl.DataFrame:
        starts_absent = pl.col(f"trait{trait_id}_episode_starts_absent")
        ends_absent = pl.col(f"trait{trait_id}_episode_ends_absent")
        gain = starts_absent & ~ends_absent
        return exposure_df.select(
            num_leaf_episodes=(starts_absent & is_leaf).sum(),
            num_leaf_gains=(gain & is_leaf).sum(),
            num_inner_episodes=(starts_absent & ~is_leaf).sum(),
            num_inner_gains=(gain & ~is_leaf).sum(),
        ).cast(pl.Int64)

    return pl.concat(map(tally_trait, range(len(trait_names)))).with_columns(
        trait_name=pl.Series(trait_names, dtype=pl.String),
    )
