import typing

import polars as pl


def _tgs_count_trait_episodes(
    exposure_df: pl.DataFrame, *, trait: int
) -> pl.DataFrame:
    """Implementation detail for `tgs_count_episodes`.

    Returns `trait`'s leaf and inner exposure episode and gain counts, as one
    row (see `tgs_count_episodes`).
    """
    starts_absent = pl.col(f"trait{trait}_starts_absent")
    gain = starts_absent & pl.col(f"trait{trait}_gain")
    is_leaf = pl.col("clade_size") == 1
    return exposure_df.select(
        (mask.cast(pl.Int64).sum().alias(name))
        for name, mask in (
            ("num_leaves", starts_absent & is_leaf),
            ("num_trait_present", gain & is_leaf),
            ("num_inner_branches", starts_absent & ~is_leaf),
            ("num_inner_gains", gain & ~is_leaf),
        )
    )


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
    columns = ["num_leaves", "num_trait_present"]
    columns += ["num_inner_branches", "num_inner_gains"]
    return pl.concat(
        [
            pl.DataFrame(schema=dict.fromkeys(columns, pl.Int64)),  # no traits
            *(  # one trait at a time, straight from the exposure table
                _tgs_count_trait_episodes(exposure_df, trait=trait)
                for trait in range(len(trait_names))
            ),
        ],
    ).select(pl.Series("trait", list(trait_names), dtype=pl.String), pl.all())
