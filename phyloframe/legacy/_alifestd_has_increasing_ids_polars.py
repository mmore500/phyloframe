import polars as pl


def alifestd_has_increasing_ids_polars(phylogeny_df: pl.DataFrame) -> bool:
    """Do offspring have larger id values than ancestors?

    Requires ancestor_id column.
    """
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        raise NotImplementedError("ancestor_id column required")

    return (
        phylogeny_df.lazy()
        .select((pl.col("id") >= pl.col("ancestor_id")).all())
        .collect()
        .item()
    )
