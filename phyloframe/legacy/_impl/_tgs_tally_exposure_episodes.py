import typing

import polars as pl


def _tgs_tally_trait_exposure_episodes(
    exposure_df: pl.DataFrame, *, trait_id: int
) -> pl.DataFrame:
    """Implementation detail for `tgs_tally_exposure_episodes`.

    Returns `trait_id`'s leaf and inner exposure episode and gain counts, as one
    row; see Output Schema.

    Output Schema
    -------------
    'num_leaf_episodes' : integer
        Leaf exposure episodes, i.e., leaf branches that started without
        the trait.
    'num_leaf_gains' : integer
        Of those, the number that gained the trait.
    'num_inner_episodes' : integer
        Inner exposure episodes, i.e., inner branches that started without
        the trait.
    'num_inner_gains' : integer
        Of those, the number that gained the trait.
    """
    starts_absent = pl.col(f"trait{trait_id}_starts_absent")
    gain = pl.col(f"trait{trait_id}_gain")  # only possible if starts absent
    is_leaf = pl.col("clade_size") == 1
    return exposure_df.select(
        mask.cast(pl.Int64).sum().alias(name)
        for name, mask in dict(
            num_leaf_episodes=starts_absent & is_leaf,
            num_leaf_gains=gain & is_leaf,
            num_inner_episodes=starts_absent & ~is_leaf,
            num_inner_gains=gain & ~is_leaf,
        ).items()
    )


def tgs_tally_exposure_episodes(
    *,
    trait_names: typing.Sequence[str],
    exposure_df: pl.DataFrame,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns one row per trait, with counts of its exposure episodes in
    `exposure_df` (see `_tgs_build_exposure_df`); see Output Schema.

    Output Schema
    -------------
    'trait' : string
        Trait name, from `trait_names`.
    'num_leaf_episodes' : integer
        Leaf exposure episodes, i.e., leaf branches that started without
        the trait.
    'num_leaf_gains' : integer
        Of those, the number that gained the trait.
    'num_inner_episodes' : integer
        Inner exposure episodes, i.e., inner branches that started without
        the trait.
    'num_inner_gains' : integer
        Of those, the number that gained the trait.
    """
    return pl.concat(
        # one trait at a time reduces peak memory
        _tgs_tally_trait_exposure_episodes(exposure_df, trait_id=trait_id)
        for trait_id in range(len(trait_names))
    ).select(pl.Series("trait", list(trait_names), dtype=pl.String), pl.all())
