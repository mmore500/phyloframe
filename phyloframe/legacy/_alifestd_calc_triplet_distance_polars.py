import typing

import polars as pl
import tqdist

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
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars


def _alifestd_make_distance_newicks_polars(
    ref: pl.DataFrame,
    cmp: pl.DataFrame,
    taxon_label_key: str,
) -> typing.Tuple[str, str]:
    """Prepare a pair of trees for tree-distance comparison.

    Collapses unifurcations, validates that leaf taxon labels match
    between the two trees, and returns Newick representations suitable
    for use with ``tqdist``.
    """
    if (
        alifestd_count_root_nodes_polars(ref) > 1
        or alifestd_count_root_nodes_polars(cmp) > 1
    ):
        raise ValueError(
            "Cannot have disjunct trees in distance calculation",
        )

    label_col = "_alifestd_distance_taxon_label"
    ref = (
        ref.with_columns(pl.col(taxon_label_key).alias(label_col))
        .pipe(alifestd_collapse_unifurcations_polars)
        .pipe(alifestd_assign_contiguous_ids_polars)
        .pipe(alifestd_mark_leaves_polars)
        .with_columns(
            pl.when(pl.col("is_leaf"))
            .then(pl.col(label_col).cast(pl.String))
            .otherwise(pl.lit(""))
            .alias(label_col),
        )
    )
    cmp = (
        cmp.with_columns(pl.col(taxon_label_key).alias(label_col))
        .pipe(alifestd_collapse_unifurcations_polars)
        .pipe(alifestd_assign_contiguous_ids_polars)
        .pipe(alifestd_mark_leaves_polars)
        .with_columns(
            pl.when(pl.col("is_leaf"))
            .then(pl.col(label_col).cast(pl.String))
            .otherwise(pl.lit(""))
            .alias(label_col),
        )
    )

    ref_leaf_labels = (
        ref.lazy()
        .filter(pl.col("is_leaf"))
        .select(pl.col(label_col).alias("label"))
        .unique()
    )
    cmp_leaf_labels = (
        cmp.lazy()
        .filter(pl.col("is_leaf"))
        .select(pl.col(label_col).alias("label"))
        .unique()
    )
    if (
        not ref_leaf_labels.join(
            cmp_leaf_labels, on="label", how="full", coalesce=False
        )
        .filter(pl.col("label").is_null() | pl.col("label_right").is_null())
        .limit(1)
        .collect()
        .is_empty()
    ):
        raise ValueError("Taxon labels must match between trees")

    empty_labels = ref_leaf_labels.filter(
        pl.col("label").str.strip_chars().str.len_chars() == 0,
    )
    if not empty_labels.limit(1).collect().is_empty():
        raise ValueError("Cannot have empty taxon labels")

    ref_newick = (
        alifestd_as_newick_polars(ref, taxon_label=label_col)
        .removeprefix("[&R]")
        .strip()
    )
    cmp_newick = (
        alifestd_as_newick_polars(cmp, taxon_label=label_col)
        .removeprefix("[&R]")
        .strip()
    )
    return ref_newick, cmp_newick


# adapted from https://github.com/mmore500/hstrat/blob/d23917cf/tests/test_hstrat/test_phylogenetic_inference/test_tree/_impl/_tree_quartet_distance.py
def alifestd_calc_triplet_distance_polars(
    ref: pl.DataFrame,
    cmp: pl.DataFrame,
    taxon_label_key: str = "taxon_label",
) -> float:
    """Calculate the triplet distance between two trees.

    Inputs must be in working asexual format: contiguous ``id`` column,
    topologically sorted rows, ``ancestor_id`` column present (no
    ``ancestor_list``).

    See Also
    --------
    alifestd_calc_triplet_distance_asexual :
        Pandas-based implementation.
    """
    ref_newick, cmp_newick = _alifestd_make_distance_newicks_polars(
        ref,
        cmp,
        taxon_label_key,
    )
    return tqdist.triplet_distance(ref_newick, cmp_newick)
