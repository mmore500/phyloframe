import polars as pl

from ._alifestd_mark_num_children_polars import (
    alifestd_mark_num_children_polars,
)
from ._alifestd_mark_roots_polars import alifestd_mark_roots_polars
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_is_strictly_bifurcating_polars(
    phylogeny_df: pl.DataFrame,
) -> bool:
    """Are all internal nodes strictly bifurcating (exactly 2 children)?"""
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if "num_children" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_mark_num_children_polars(phylogeny_df)

    if "is_root" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_mark_roots_polars(phylogeny_df)

    # Internal (non-leaf) non-root nodes must have exactly 2 children.
    # Also, root nodes that are not leaves must have exactly 2 children.
    # Equivalently: every node with children must have exactly 2.
    return (
        phylogeny_df.lazy()
        .filter(pl.col("num_children") > 0)
        .select((pl.col("num_children") == 2).all())
        .collect()
        .item()
    )
