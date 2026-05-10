import typing

import polars as pl

from ._alifestd_as_newick_polars import alifestd_as_newick_polars
from ._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from ._alifestd_collapse_unifurcations_polars import (
    alifestd_collapse_unifurcations_polars,
)
from ._alifestd_count_root_nodes_polars import (
    alifestd_count_root_nodes_polars,
)
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars
from ._alifestd_topological_sort_polars import (
    alifestd_topological_sort_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def _normalize(phylogeny_df: pl.DataFrame) -> pl.DataFrame:
    if isinstance(phylogeny_df, pl.LazyFrame):
        phylogeny_df = phylogeny_df.collect()
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    if "ancestor_list" in phylogeny_df.collect_schema().names():
        phylogeny_df = phylogeny_df.drop("ancestor_list")
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        phylogeny_df = alifestd_topological_sort_polars(phylogeny_df)
        phylogeny_df = alifestd_assign_contiguous_ids_polars(phylogeny_df)
    return phylogeny_df


def _alifestd_make_distance_newicks_polars(
    ref: pl.DataFrame,
    cmp: pl.DataFrame,
    taxon_label_key: str,
    *,
    caller_name: str,
) -> typing.Tuple[str, str]:
    """Prepare a pair of trees for tree-distance comparison.

    Collapses unifurcations, validates that leaf taxon labels match
    between the two trees, and returns Newick representations suitable
    for use with ``tqdist``.
    """
    ref = alifestd_mark_leaves_polars(
        _normalize(alifestd_collapse_unifurcations_polars(_normalize(ref))),
    )
    cmp = alifestd_mark_leaves_polars(
        _normalize(alifestd_collapse_unifurcations_polars(_normalize(cmp))),
    )

    ref = ref.with_columns(
        pl.when(pl.col("is_leaf"))
        .then(pl.col(taxon_label_key).cast(pl.String))
        .otherwise(pl.lit(""))
        .alias(taxon_label_key),
    )
    cmp = cmp.with_columns(
        pl.when(pl.col("is_leaf"))
        .then(pl.col(taxon_label_key).cast(pl.String))
        .otherwise(pl.lit(""))
        .alias(taxon_label_key),
    )

    ref_labels = {
        *ref.lazy()
        .filter(pl.col("is_leaf"))
        .select(taxon_label_key)
        .collect()
        .to_series()
        .to_list()
    }
    cmp_labels = {
        *cmp.lazy()
        .filter(pl.col("is_leaf"))
        .select(taxon_label_key)
        .collect()
        .to_series()
        .to_list()
    }

    if ref_labels != cmp_labels:
        raise ValueError("Taxon labels must match between trees")
    for taxon_label in ref_labels:
        if isinstance(taxon_label, str) and not taxon_label.strip():
            raise ValueError("Cannot have empty taxon labels")

    if (
        alifestd_count_root_nodes_polars(ref) > 1
        or alifestd_count_root_nodes_polars(cmp) > 1
    ):
        raise ValueError(
            f"Cannot have disjunct trees in `{caller_name}`",
        )

    ref_newick = (
        alifestd_as_newick_polars(ref, taxon_label=taxon_label_key)
        .removeprefix("[&R]")
        .strip()
    )
    cmp_newick = (
        alifestd_as_newick_polars(cmp, taxon_label=taxon_label_key)
        .removeprefix("[&R]")
        .strip()
    )
    return ref_newick, cmp_newick
