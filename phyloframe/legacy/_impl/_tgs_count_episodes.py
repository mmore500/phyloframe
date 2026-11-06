import typing

import polars as pl


def tgs_count_episodes(
    *,
    trait_names: typing.Sequence[str],
    exposure_df: pl.DataFrame,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns one row per trait, with counts of its exposure episodes in
    `exposure_df` (see `tgs_calc_exposure_df`); see Output Schema.

    Output Schema
    -------------
    'trait' : string
        Trait name, from `trait_names`.
    'num_leaves' : integer
        Leaf exposure episodes, i.e., leaf branches that started without
        the trait.
    'num_trait_present' : integer
        Of those, the number that gained the trait.
    'num_inner_branches' : integer
        Inner exposure episodes, i.e., inner branches that started without
        the trait.
    'num_inner_gains' : integer
        Of those, the number that gained the trait.
    """
    is_leaf = pl.col("clade_size") == 1
    return pl.concat(
        [
            # one trait at a time, straight from the exposure table
            exposure_df.select(
                (starts_absent & is_leaf).sum().alias("num_leaves"),
                (gain & is_leaf).sum().alias("num_trait_present"),
                (starts_absent & ~is_leaf).sum().alias("num_inner_branches"),
                (gain & ~is_leaf).sum().alias("num_inner_gains"),
            )
            for starts_absent, gain in (
                (
                    pl.col(f"trait{trait}_starts_absent"),
                    pl.col(f"trait{trait}_starts_absent")
                    & pl.col(f"trait{trait}_gain"),
                )
                for trait in range(len(trait_names))
            )
        ],
    ).select(
        pl.Series("trait", list(trait_names), dtype=pl.String),
        pl.all().cast(pl.Int64),
    )
