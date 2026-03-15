import polars as pl


def alifestd_has_compact_ids_polars(phylogeny_df: pl.DataFrame) -> bool:
    """Are id values between 0 and `len(phylogeny_df)`, in any order?"""
    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return True

    return (
        phylogeny_df.lazy()
        .select(pl.col("id").max() == pl.len() - 1)
        .collect()
        .item()
    )
