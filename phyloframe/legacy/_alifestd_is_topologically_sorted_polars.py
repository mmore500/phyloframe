import polars as pl

from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_is_topologically_sorted_polars(
    phylogeny_df: pl.DataFrame,
) -> bool:
    """Are all organisms listed after members of their `ancestor_list`?"""

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return True

    # Fast path: if ids are monotonically non-decreasing, can use simple
    # comparison
    ids_sorted = (
        phylogeny_df.lazy()
        .select(
            pl.col("id").diff().drop_nulls().ge(0).all(),
        )
        .collect()
        .item()
    )
    if ids_sorted:
        return (
            phylogeny_df.lazy()
            .select((pl.col("ancestor_id") <= pl.col("id")).all())
            .collect()
            .item()
        )

    # Slow path: build position map and check ancestor positions
    position_map = phylogeny_df.lazy().with_row_index("_pos")
    return (
        position_map.join(
            position_map.select(
                pl.col("id").alias("ancestor_id"),
                pl.col("_pos").alias("_ancestor_pos"),
            ),
            on="ancestor_id",
            how="left",
        )
        .select(
            (
                pl.col("_ancestor_pos").is_not_null()
                & (pl.col("_ancestor_pos") <= pl.col("_pos"))
            ).all()
        )
        .collect()
        .item()
    )
