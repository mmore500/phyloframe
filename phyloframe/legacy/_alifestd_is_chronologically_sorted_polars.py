import polars as pl


def alifestd_is_chronologically_sorted_polars(
    phylogeny_df: pl.DataFrame,
    how: str = "origin_time",
) -> bool:
    """Do rows appear in chronological order?

    Defaults to `origin_time`.
    """
    return how in phylogeny_df.lazy().collect_schema().names() and (
        phylogeny_df.lazy()
        .select(
            pl.col(how).diff().drop_nulls().ge(0).all(),
        )
        .collect()
        .item()
    )
