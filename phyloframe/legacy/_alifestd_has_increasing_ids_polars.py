import polars as pl

from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_has_increasing_ids_polars(phylogeny_df: pl.DataFrame) -> bool:
    """Do offspring have larger id values than ancestors?

    Requires ancestor_id column.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    return (
        phylogeny_df.lazy()
        .select((pl.col("id") >= pl.col("ancestor_id")).all())
        .collect()
        .item()
    )
