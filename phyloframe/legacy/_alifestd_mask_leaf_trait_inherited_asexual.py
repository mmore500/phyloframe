import numpy as np
import pandas as pd

from .._auxlib._experimental import experimental
from .._auxlib._jit import jit
from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_leaves import _alifestd_mark_leaves_asexual_fast_path
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_mark_roots import _alifestd_mark_roots_asexual_fast_path
from ._alifestd_mask_monomorphic_clades_asexual import (
    _alifestd_mask_monomorphic_clades_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_mask_leaf_trait_inherited_asexual_nearest_branching_ancestors(
    ancestor_ids: np.ndarray,
    is_branching: np.ndarray,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_mask_leaf_trait_inherited_asexual`.

    Returns, for each node, its parent, skipping over unifurcations: the
    nearest ancestor that has at least two children (`is_branching`) or is
    a root. Roots map to themselves. Assumes contiguous, topologically
    sorted ids.
    """
    res = ancestor_ids.copy()
    for idx, ancestor_id in enumerate(ancestor_ids):
        is_root = ancestor_ids[ancestor_id] == ancestor_id
        if (
            ancestor_id != idx
            and not is_branching[ancestor_id]
            and not is_root
        ):
            res[idx] = res[ancestor_id]  # computed already, as sorted
    return res


def _alifestd_mask_leaf_trait_inherited_asexual_fast_path(
    ancestor_ids: np.ndarray,
    trait_mask: np.ndarray,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_mask_leaf_trait_inherited_asexual`."""
    is_leaf = _alifestd_mark_leaves_asexual_fast_path(ancestor_ids)
    is_present = is_leaf & trait_mask
    # a clade is monomorphic with respect to leaf trait values if all its
    # leaves share one value
    is_monomorphic = _alifestd_mask_monomorphic_clades_asexual_fast_path(
        ancestor_ids, is_leaf.copy(), is_present.copy()
    )
    # a trait-carrying leaf under a monomorphic parent shares the trait with
    # every leaf of its parent's clade; unifurcations are bypassed, as a
    # unifurcation's clade has the same leaves as its child's
    parents = _alifestd_mask_leaf_trait_inherited_asexual_nearest_branching_ancestors(
        ancestor_ids,
        _alifestd_mark_num_children_asexual_fast_path(ancestor_ids) >= 2,
    )
    is_root = _alifestd_mark_roots_asexual_fast_path(ancestor_ids)
    return is_present & is_monomorphic[parents] & ~is_root


@experimental
def alifestd_mask_leaf_trait_inherited_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "alifestd_mask_leaf_trait_inherited_asexual",
    trait_mask: np.ndarray,
) -> pd.DataFrame:
    """Mark leaves whose trait is shared by every leaf descending from their
    parent, so that the trait need not have arisen on the leaf's own pendant
    branch.

    Under parsimony, such a leaf's trait was gained above its parent node;
    other leaves with the trait gained it on their pendant branch. Root
    leaves and leaves without the trait are marked False. Unifurcations are
    bypassed, so a leaf's parent is its nearest branching ancestor.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        DataFrame containing the phylogeny, including an `ancestor_id` or
        `ancestor_list` column.
    mutate : bool, default False
        If False, operates on a copy of `phylogeny_df`; if True, modifies
        `phylogeny_df` in place (but still returns it).
    mark_as : str, default "alifestd_mask_leaf_trait_inherited_asexual"
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

    phylogeny_df[
        mark_as
    ] = _alifestd_mask_leaf_trait_inherited_asexual_fast_path(
        phylogeny_df["ancestor_id"].to_numpy(),
        trait_mask,
    )
    return phylogeny_df
