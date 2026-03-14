import polars as pl


def alifestd_has_multiple_roots_polars(
    phylogeny_df: pl.DataFrame,
) -> bool:
    """Does the phylogeny have two or more root organisms?"""
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        raise NotImplementedError("ancestor_id column required")

    return (
        len(
            phylogeny_df.lazy()
            .filter(pl.col("ancestor_id") == pl.col("id"))
            .limit(2)
            .collect()
        )
        >= 2
    )
