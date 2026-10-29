import numpy as np
import pandas as pd

from .._auxlib._jit import jit
from .._auxlib._warn_experimental import warn_experimental
from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_leaves import _alifestd_mark_leaves_asexual_fast_path
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_mask_monomorphic_clades_asexual import (
    _alifestd_mask_monomorphic_clades_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mask_trait_singletons_asexual_stem_node_ids(
    ancestor_ids: np.ndarray,
    is_branching: np.ndarray,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_mask_trait_singletons_asexual`.

    Returns, for each node, its stem node: the nearest ancestor that has at
    least two children (`is_branching`), where the node's lineage splits
    from its sister lineages, skipping over unifurcations. A node without
    one (e.g., a root) is its own stem node. Assumes contiguous,
    topologically sorted ids.
    """
    res = np.empty_like(ancestor_ids)
    for idx, ancestor_id in enumerate(ancestor_ids):
        if ancestor_id == idx:  # root, without ancestors
            res[idx] = idx
        elif is_branching[ancestor_id]:
            res[idx] = ancestor_id
        elif res[ancestor_id] != ancestor_id:  # computed already, as sorted
            res[idx] = res[ancestor_id]
        else:  # no stem node above a unifurcating ancestor
            res[idx] = idx
    return res


def _alifestd_mask_trait_singletons_asexual_fast_path(
    ancestor_ids: np.ndarray,
    trait_mask: np.ndarray,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_mask_trait_singletons_asexual`."""
    is_leaf = _alifestd_mark_leaves_asexual_fast_path(ancestor_ids)
    is_present = is_leaf & trait_mask
    # a clade is monomorphic with respect to leaf trait values if all its
    # leaves share one value
    is_monomorphic = _alifestd_mask_monomorphic_clades_asexual_fast_path(
        ancestor_ids, is_leaf.copy(), is_present.copy()
    )
    # a trait-carrying leaf whose stem node's clade is polymorphic does not
    # share the trait with every leaf of its sister lineages, so gained it on
    # its own pendant branch; unifurcations are bypassed, as a unifurcation's
    # clade has the same leaves as its child's; a leaf without a stem node
    # (e.g., a root leaf) is its own, so is never marked, as there is no
    # other leaf to compare against
    stem_node_ids = _alifestd_mask_trait_singletons_asexual_stem_node_ids(
        ancestor_ids,
        _alifestd_mark_num_children_asexual_fast_path(ancestor_ids) >= 2,
    )
    return is_present & ~is_monomorphic[stem_node_ids]


@warn_experimental
def alifestd_mask_trait_singletons_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "alifestd_mask_trait_singletons_asexual",
    trait_mask: np.ndarray,
) -> pd.DataFrame:
    """Mark trait singletons: leaves with a trait that is not shared by
    every leaf descending from their parent, i.e., all-carrier clades of a
    single leaf, which gained the trait on their own pendant branch.

    Under parsimony, such a leaf's trait arose on its pendant branch; a
    leaf whose trait is shared by every leaf of its parent's clade could
    have gained it above its parent node, so is marked False. Root leaves
    and leaves without the trait are also marked False. Unifurcations are
    bypassed, so a leaf's parent is its stem node, the nearest ancestor
    with at least two children; leaves without one have no other leaf to
    compare against, so are marked False.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        DataFrame containing the phylogeny, including an `ancestor_id` or
        `ancestor_list` column.
    mutate : bool, default False
        If False, operates on a copy of `phylogeny_df`; if True, modifies
        `phylogeny_df` in place (but still returns it).
    mark_as : str, default "alifestd_mask_trait_singletons_asexual"
        Column name for the mask.
    trait_mask : np.ndarray
        Boolean array marking nodes that have the trait, aligned with
        `phylogeny_df.index`, with boolean dtype. Values at inner nodes
        are ignored.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not in working format (contiguous ids,
        topologically sorted; see `alifestd_to_working_format`), or if
        `trait_mask` does not have boolean dtype.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    trait_mask = np.asarray(trait_mask)
    if trait_mask.dtype != np.bool_:
        raise NotImplementedError(
            f"trait_mask must have boolean dtype, got {trait_mask.dtype}",
        )
    if not alifestd_is_working_format_asexual(phylogeny_df, mutate=True):
        raise NotImplementedError(
            "current implementation requires phylogeny_df in working format",
        )

    phylogeny_df[mark_as] = _alifestd_mask_trait_singletons_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(),
        trait_mask,
    )
    return phylogeny_df
