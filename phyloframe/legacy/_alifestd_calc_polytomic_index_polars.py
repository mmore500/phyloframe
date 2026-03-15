import polars as pl

from ._alifestd_count_leaf_nodes_polars import alifestd_count_leaf_nodes_polars
from ._alifestd_count_root_nodes_polars import alifestd_count_root_nodes_polars
from ._alifestd_count_unifurcations_polars import (
    alifestd_count_unifurcations_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_calc_polytomic_index_polars(
    phylogeny_df: pl.DataFrame,
) -> int:
    """Count how many fewer inner nodes are contained in phylogeny than
    expected if strictly bifurcating.

    Excludes unifurcations from calculation.
    """
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    num_leaf_nodes = alifestd_count_leaf_nodes_polars(phylogeny_df)
    num_root_nodes = alifestd_count_root_nodes_polars(phylogeny_df)
    expected_rows_if_bifurcating = max(2 * num_leaf_nodes - num_root_nodes, 0)
    num_unifurcations = alifestd_count_unifurcations_polars(phylogeny_df)
    num_rows = phylogeny_df.lazy().select(pl.len()).collect().item()
    num_non_unifurcating_rows = num_rows - num_unifurcations
    res = expected_rows_if_bifurcating - num_non_unifurcating_rows
    assert 0 <= res < max(expected_rows_if_bifurcating, 1)
    return res
