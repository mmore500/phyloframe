import polars as pl


def alifestd_is_sexual_polars(phylogeny_df: pl.DataFrame) -> bool:
    """Do any organisms in the phylogeny have more than one immediate
    ancestor?"""
    return (
        phylogeny_df.lazy()
        .select(pl.col("ancestor_list").cast(pl.Utf8).str.contains(",").any())
        .collect()
        .item()
    )
