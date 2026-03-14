import polars as pl


def alifestd_make_ancestor_id_col_polars(
    ids: pl.Series, ancestor_lists: pl.Series
) -> pl.Series:
    """Translate ancestor ids from a column of singleton `ancestor_list`s
    into a pure-integer series representation.

    Each organism must have one or zero ancestors (i.e., asexual data).
    In the returned series, ancestor id will be assigned to own id for
    no-ancestor organisms.
    """
    df = pl.DataFrame({"ids": ids, "ancestor_lists": ancestor_lists})
    result = df.select(
        pl.col("ancestor_lists")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"\[none\]", "[-1]")
        .str.replace_all(r"\[\]", "[-1]")
        .str.strip_chars("[]")
        .cast(pl.Int64)
        .alias("ancestor_id"),
        pl.col("ids"),
    ).select(
        pl.when(pl.col("ancestor_id") == -1)
        .then(pl.col("ids"))
        .otherwise(pl.col("ancestor_id"))
        .alias("ancestor_id"),
    )
    return result["ancestor_id"]
