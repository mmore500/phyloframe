import typing

import polars as pl

from ._tgs_unpivot_exposure_episodes import tgs_unpivot_exposure_episodes


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
    'num_clades' : integer
        Inner exposure episodes, i.e., inner branches that started without
        the trait.
    'num_clade_gains' : integer
        Of those, the number that gained the trait.
    """
    episode_df = tgs_unpivot_exposure_episodes(exposure_df)
    is_leaf = exposure_df["clade_size"].gather(episode_df["branch"]) == 1
    counts_df = (
        episode_df.with_columns(is_leaf=is_leaf)
        .group_by("trait")
        .agg(
            num_leaves=pl.col("is_leaf").cast(pl.Int64).sum(),
            num_trait_present=(pl.col("gain") & pl.col("is_leaf"))
            .cast(pl.Int64)
            .sum(),
            num_clades=(~pl.col("is_leaf")).cast(pl.Int64).sum(),
            num_clade_gains=(pl.col("gain") & ~pl.col("is_leaf"))
            .cast(pl.Int64)
            .sum(),
        )
    )
    # every trait, including those without episodes
    return (
        pl.DataFrame(
            {
                "trait": list(trait_names),
                "trait_index": range(len(trait_names)),
            },
            schema_overrides={"trait_index": pl.Int64},
        )
        .join(
            counts_df.rename({"trait": "trait_index"}),
            on="trait_index",
            how="left",
            maintain_order="left",
        )
        .drop("trait_index")
        .fill_null(0)
    )
