import warnings

import numpy as np
import pandas as pd

from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_leaves import _alifestd_mark_leaves_asexual_fast_path
from ._alifestd_mask_monomorphic_clades_asexual import (
    _alifestd_mask_monomorphic_clades_asexual_fast_path,
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
    # every leaf of its parent's clade
    is_root = ancestor_ids == np.arange(len(ancestor_ids))
    return is_present & is_monomorphic[ancestor_ids] & ~is_root


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

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not in working format (contiguous ids,
        topologically sorted); see `alifestd_to_working_format`.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)
    trait_mask = _alifestd_mask_leaf_trait_inherited_asexual_coerce_mask(
        trait_mask
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
