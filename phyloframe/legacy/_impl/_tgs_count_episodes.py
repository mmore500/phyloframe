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

    Returns one row per trait, with column `trait` (from `trait_names`) and
    counts, from `exposure_df` (see `tgs_calc_exposure_df`), of leaf
    exposure episodes, i.e., leaf branches starting without the trait,
    `num_leaves`, and of their gains, `num_trait_present`, and the same for
    inner exposure episodes, `num_clades` and `num_clade_gains`.
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
