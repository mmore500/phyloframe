import warnings

import numpy as np
import pandas as pd

from .._auxlib._jit import jit
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_leaves import alifestd_mark_leaves
from ._alifestd_mask_monomorphic_clades_asexual import (
    alifestd_mask_monomorphic_clades_asexual,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_mask_leaf_trait_inherited_asexual_coerce_mask(
    trait_mask: np.ndarray, name: str = "trait_mask"
) -> np.ndarray:
    """Implementation detail for
    `alifestd_mask_leaf_trait_inherited_asexual` and
    `alifestd_estimate_trait_gain_selection_asexual`.

    Casts `trait_mask` to bool, treating null values as False with a
    warning, matching polars implementations.
    """
    is_null = pd.isna(np.asarray(trait_mask))
    if is_null.any():
        warnings.warn(
            f"{name} contains null values, "
            "which will be treated as trait absent (False).",
        )
    return np.where(is_null, False, trait_mask).astype(bool)


@jit(nopython=True)
def _alifestd_mask_leaf_trait_inherited_asexual_fast_path(
    ancestor_ids: np.ndarray,
    trait_mask: np.ndarray,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_mask_leaf_trait_inherited_asexual`."""
    num_nodes = len(ancestor_ids)
    is_leaf = np.ones(num_nodes, dtype=np.bool_)
    for idx, ancestor_id in enumerate(ancestor_ids):
        if ancestor_id != idx:
            is_leaf[ancestor_id] = False

    # bottom-up pass: does every leaf descending from each node have trait?
    all_present = trait_mask | ~is_leaf
    for idx_r, ancestor_id in enumerate(ancestor_ids[::-1]):
        idx = num_nodes - 1 - idx_r
        if ancestor_id != idx:  # exclude genesis cases
            all_present[ancestor_id] &= all_present[idx]

    res = np.zeros(num_nodes, dtype=np.bool_)
    for idx, ancestor_id in enumerate(ancestor_ids):
        res[idx] = (
            is_leaf[idx]
            and trait_mask[idx]
            and ancestor_id != idx
            and all_present[ancestor_id]
        )
    return res


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
    leaves and leaves without the trait are marked False.

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
        `phylogeny_df.index`. Values at inner nodes are ignored. Null
        values are treated as trait absent, with a warning.

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    A topological sort will be applied if `phylogeny_df` is not
    topologically sorted. Dataframe reindexing (e.g., df.index) may be
    applied.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    trait_mask = _alifestd_mask_leaf_trait_inherited_asexual_coerce_mask(
        trait_mask
    )
    if alifestd_is_working_format_asexual(phylogeny_df, mutate=True):
        phylogeny_df[
            mark_as
        ] = _alifestd_mask_leaf_trait_inherited_asexual_fast_path(
            phylogeny_df["ancestor_id"].to_numpy(),
            trait_mask,
        )
        return phylogeny_df

    phylogeny_df[mark_as] = trait_mask

    if "is_leaf" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_leaves(phylogeny_df, mutate=True)

    # leaves only, so inner node values are ignored
    phylogeny_df[mark_as] &= phylogeny_df["is_leaf"]

    # a clade is monomorphic with respect to leaf trait values if all its
    # leaves share one value
    temp_col = "alifestd_mask_leaf_trait_inherited_asexual_monomorphic"
    phylogeny_df = alifestd_mask_monomorphic_clades_asexual(
        phylogeny_df,
        mutate=True,
        mark_as=temp_col,
        trait_mask=phylogeny_df["is_leaf"].to_numpy(),
        trait_values=phylogeny_df[mark_as].to_numpy(),
    )

    if alifestd_has_contiguous_ids(phylogeny_df):
        phylogeny_df.reset_index(drop=True, inplace=True)
    else:
        phylogeny_df.index = phylogeny_df["id"]

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    is_root = ancestor_ids == phylogeny_df["id"].to_numpy()
    # a trait-carrying leaf under a monomorphic parent shares the trait with
    # every leaf of its parent's clade
    parent_monomorphic = phylogeny_df.loc[ancestor_ids, temp_col].to_numpy()
    phylogeny_df[mark_as] &= parent_monomorphic & ~is_root

    phylogeny_df.drop(columns=[temp_col], inplace=True)
    return phylogeny_df
