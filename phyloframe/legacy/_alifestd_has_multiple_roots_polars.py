import polars as pl


def alifestd_has_multiple_roots_polars(
    phylogeny_df: pl.DataFrame,
) -> bool:
    """Does the phylogeny have two or more root organisms?"""
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    return (
        phylogeny_df.lazy()
        .filter(pl.col("ancestor_id") == pl.col("id"))
        .limit(2)
        .select(pl.len())
        .collect()
        .item()
    ) >= 2
