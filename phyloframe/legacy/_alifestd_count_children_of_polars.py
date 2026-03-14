import polars as pl

from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_count_children_of_polars(
    phylogeny_df: pl.DataFrame,
    parent: int,
) -> int:
    """How many taxa are direct descendants of the given parent?"""
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if not (
        phylogeny_df.lazy()
        .select((pl.col("id") == parent).any())
        .collect()
        .item()
    ):
        raise ValueError(f"Parent {parent} not found in phylogeny dataframe.")

    return (
        phylogeny_df.lazy()
        .select(
            (
                (pl.col("ancestor_id") == parent)
                & (pl.col("id") != pl.col("ancestor_id"))
            ).sum()
        )
        .collect()
        .item()
    )
