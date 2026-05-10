import typing

import pandas as pd
import tqdist

from ._alifestd_as_newick_asexual import alifestd_as_newick_asexual
from ._alifestd_collapse_unifurcations import alifestd_collapse_unifurcations
from ._alifestd_count_root_nodes import alifestd_count_root_nodes
from ._alifestd_mark_leaves import alifestd_mark_leaves


def _alifestd_make_distance_newicks_asexual(
    ref: pd.DataFrame,
    cmp: pd.DataFrame,
    taxon_label_key: str,
) -> typing.Tuple[str, str]:
    """Prepare a pair of trees for tree-distance comparison.

    Collapses unifurcations, validates that leaf taxon labels match
    between the two trees, and returns Newick representations suitable
    for use with ``tqdist``.
    """
    if (
        alifestd_count_root_nodes(ref) > 1
        or alifestd_count_root_nodes(cmp) > 1
    ):
        raise ValueError(
            "Cannot have disjunct trees in distance calculation",
        )

    label_col = "_alifestd_distance_taxon_label"
    ref = ref.copy()
    cmp = cmp.copy()
    ref[label_col] = ref[taxon_label_key].astype(str)
    cmp[label_col] = cmp[taxon_label_key].astype(str)
    ref = ref.pipe(alifestd_collapse_unifurcations, mutate=True).pipe(
        alifestd_mark_leaves, mutate=True
    )
    cmp = cmp.pipe(alifestd_collapse_unifurcations, mutate=True).pipe(
        alifestd_mark_leaves, mutate=True
    )
    ref.loc[~ref["is_leaf"], label_col] = ""
    cmp.loc[~cmp["is_leaf"], label_col] = ""

    ref_labels = {*ref[label_col][ref["is_leaf"]]}
    cmp_labels = {*cmp[label_col][cmp["is_leaf"]]}

    if ref_labels != cmp_labels:
        raise ValueError("Taxon labels must match between trees")
    for taxon_label in ref_labels:
        if not taxon_label.strip():
            raise ValueError("Cannot have empty taxon labels")

    ref_newick = (
        alifestd_as_newick_asexual(ref, taxon_label=label_col)
        .removeprefix("[&R]")
        .strip()
    )
    cmp_newick = (
        alifestd_as_newick_asexual(cmp, taxon_label=label_col)
        .removeprefix("[&R]")
        .strip()
    )
    return ref_newick, cmp_newick


# adapted from https://github.com/mmore500/hstrat/blob/d23917cf/tests/test_hstrat/test_phylogenetic_inference/test_tree/_impl/_tree_quartet_distance.py
def alifestd_calc_triplet_distance_asexual(
    ref: pd.DataFrame,
    cmp: pd.DataFrame,
    taxon_label_key: str = "taxon_label",
) -> float:
    """Calculate the triplet distance between two trees."""
    ref_newick, cmp_newick = _alifestd_make_distance_newicks_asexual(
        ref,
        cmp,
        taxon_label_key,
    )
    return tqdist.triplet_distance(ref_newick, cmp_newick)
