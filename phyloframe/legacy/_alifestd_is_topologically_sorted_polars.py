import logging
import os

import polars as pl


def alifestd_is_topologically_sorted_polars(
    phylogeny_df: pl.DataFrame,
) -> bool:
    """Are all organisms listed after members of their `ancestor_list`?"""

    if os.environ.get(
        "PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_IS_TOPOLOGICALLY_SORTED_POLARS",
    ):
        logging.info(
            "- alifestd_is_topologically_sorted_polars: bypassing check, "
            "PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_IS_TOPOLOGICALLY_SORTED_POLARS is set",
        )
        return True

    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        raise NotImplementedError("ancestor_id column required")

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
